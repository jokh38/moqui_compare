#include "mqi_work_queue.hpp"
#include <algorithm>
#include <cstring>

namespace mqi {

CUDA_HOST
void initialize_work_queue(work_queue_t *queue, work_item_t *work_items,
                           const uint32_t total_work_items,
                           const uint32_t min_work_size,
                           const uint32_t max_work_size,
                           const uint32_t work_stealing_enabled,
                           const uint32_t dynamic_rebalance,
                           const float load_balance_threshold) {
  if (!queue || !work_items) {
    return;
  }

  // Initialize work item array
  queue->work_items = work_items;
  queue->total_work_items = total_work_items;
  queue->work_item_stride = sizeof(work_item_t);

  // Initialize atomic counters
  queue->next_work_item.store(0);
  queue->completed_work.store(0);
  queue->active_workers.store(0);
  queue->work_steals.store(0);
  queue->balance_operations.store(0);

  // Set configuration parameters
  queue->min_work_size = min_work_size;
  queue->max_work_size = max_work_size;
  queue->work_stealing_enabled = work_stealing_enabled;
  queue->dynamic_rebalance = dynamic_rebalance;
  queue->load_balance_threshold =
      std::max(0.0f, std::min(1.0f, load_balance_threshold));
}

CUDA_HOST
work_item_t get_next_work_item(work_queue_t *queue) {
  if (!queue || !queue->work_items) {
    // Return empty work item if queue is invalid
    work_item_t empty_item = {0, 0, 0, 0, 0, 0.0f};
    return empty_item;
  }

  // Atomically get next work item index
  uint32_t work_index = queue->next_work_item.fetch_add(1);

  if (work_index >= queue->total_work_items) {
    // No more work items available
    work_item_t empty_item = {0, 0, 0, 0, 0, 0.0f};
    return empty_item;
  }

  // Increment active workers counter
  queue->active_workers.fetch_add(1);

  return queue->work_items[work_index];
}

CUDA_HOST
work_item_t steal_work(work_queue_t *queue) {
  if (!queue || !queue->work_stealing_enabled || !queue->work_items) {
    work_item_t empty_item = {0, 0, 0, 0, 0, 0.0f};
    return empty_item;
  }

  // Simple work stealing - try to get work from the end
  uint32_t current_next = queue->next_work_item.load();

  // Check if there's work to steal
  if (current_next >= queue->total_work_items) {
    work_item_t empty_item = {0, 0, 0, 0, 0, 0.0f};
    return empty_item;
  }

  // Try to steal a work item
  uint32_t steal_index = queue->total_work_items - 1;

  // Host-side work stealing
  uint32_t expected = current_next;
  if (queue->next_work_item.compare_exchange_strong(expected, steal_index)) {
    queue->work_steals.fetch_add(1);
    queue->active_workers.fetch_add(1);
    return queue->work_items[steal_index];
  }

  // Work stealing failed
  work_item_t empty_item = {0, 0, 0, 0, 0, 0.0f};
  return empty_item;
}

CUDA_HOST
void mark_work_completed(work_queue_t *queue, uint32_t work_item_id) {
  if (!queue) {
    return;
  }

  // Increment completed work counter
  queue->completed_work.fetch_add(1);
  queue->active_workers.fetch_sub(1);
}

CUDA_HOST
bool is_work_queue_empty(const work_queue_t *queue) {
  if (!queue) {
    return true;
  }

  uint32_t next_work = queue->next_work_item.load();
  uint32_t completed = queue->completed_work.load();

  return (next_work >= queue->total_work_items) &&
         (completed >= queue->total_work_items);
}

CUDA_HOST
void get_work_queue_stats(const work_queue_t *queue, uint32_t &completed_work,
                          uint32_t &active_workers, uint32_t &work_steals,
                          uint32_t &balance_ops) {
  if (!queue) {
    completed_work = 0;
    active_workers = 0;
    work_steals = 0;
    balance_ops = 0;
    return;
  }

#if defined(__CUDACC__)
  completed_work = atomic_load(&queue->completed_work);
  active_workers = atomic_load(&queue->active_workers);
  work_steals = atomic_load(&queue->work_steals);
  balance_ops = atomic_load(&queue->balance_operations);
#else
  completed_work = queue->completed_work.load();
  active_workers = queue->active_workers.load();
  work_steals = queue->work_steals.load();
  balance_ops = queue->balance_operations.load();
#endif
}

CUDA_HOST
void reset_work_queue(work_queue_t *queue) {
  if (!queue) {
    return;
  }

  // Reset atomic counters to zero
  queue->next_work_item.store(0);
  queue->completed_work.store(0);
  queue->active_workers.store(0);
  queue->work_steals.store(0);
  queue->balance_operations.store(0);
}

CUDA_HOST
void create_particle_work_items(work_queue_t *queue,
                                const uint32_t num_particles,
                                const uint32_t histories_per_particle,
                                const float base_weight) {
  if (!queue || !queue->work_items) {
    return;
  }

  uint32_t optimal_work_size = calculate_optimal_work_size(
      num_particles, 1024, queue->min_work_size, queue->max_work_size);

  uint32_t num_work_items =
      (num_particles + optimal_work_size - 1) / optimal_work_size;
  num_work_items = std::min(num_work_items, queue->total_work_items);

  for (uint32_t i = 0; i < num_work_items; ++i) {
    uint32_t start_particle = i * optimal_work_size;
    uint32_t end_particle =
        std::min(start_particle + optimal_work_size, num_particles);

    queue->work_items[i].particle_id = start_particle;
    queue->work_items[i].start_history = 0;
    queue->work_items[i].end_history = histories_per_particle;
    queue->work_items[i].priority = 0;
    queue->work_items[i].work_size = end_particle - start_particle;
    queue->work_items[i].weight =
        base_weight * static_cast<float>(queue->work_items[i].work_size);
  }

  // Update queue to reflect actual number of work items
  queue->total_work_items = num_work_items;
}

CUDA_HOST
void balance_work_load(work_queue_t *queue) {
  if (!queue || !queue->dynamic_rebalance) {
    return;
  }

  // Simple load balancing - redistribute work if imbalance is detected
  uint32_t active_workers = queue->active_workers.load();
  uint32_t completed_work = queue->completed_work.load();

  // Calculate load balance efficiency
  float efficiency =
      (completed_work > 0)
          ? static_cast<float>(completed_work) /
                static_cast<float>(active_workers + completed_work)
          : 1.0f;

  // If efficiency is below threshold, trigger rebalancing
  if (efficiency < queue->load_balance_threshold) {
    queue->balance_operations.fetch_add(1);
  }
}

CUDA_HOST
uint32_t calculate_optimal_work_size(const uint32_t total_work,
                                     const uint32_t num_threads,
                                     const uint32_t min_size,
                                     const uint32_t max_size) {
  if (total_work == 0 || num_threads == 0) {
    return min_size;
  }

  // Calculate base work size per thread
  uint32_t base_size = (total_work + num_threads - 1) / num_threads;

  // Clamp to min/max bounds
  uint32_t optimal_size = std::max(min_size, std::min(max_size, base_size));

  // Ensure we don't create more work items than work items
  if (optimal_size > total_work) {
    optimal_size = total_work;
  }

  return optimal_size;
}

CUDA_HOST
float calculate_load_balance_efficiency(const work_queue_t *queue) {
  if (!queue) {
    return 0.0f;
  }

  uint32_t active = queue->active_workers.load();
  uint32_t completed = queue->completed_work.load();

  uint32_t total = active + completed;
  if (total == 0) {
    return 1.0f;
  }

  return static_cast<float>(completed) / static_cast<float>(total);
}

CUDA_HOST
uint32_t get_thread_local_work_item_id(const work_queue_t *queue) {
  if (!queue) {
    return 0;
  }

  return queue->next_work_item.load();
}

CUDA_HOST
void update_thread_work_statistics(work_queue_t *queue, uint32_t work_done,
                                   uint32_t time_spent) {
  if (!queue) {
    return;
  }

  // This function can be extended to track more detailed statistics
  // For now, we just mark work as completed
  for (uint32_t i = 0; i < work_done; ++i) {
    mark_work_completed(queue,
                        0); // work_item_id can be made more sophisticated
  }
}

} // namespace mqi
