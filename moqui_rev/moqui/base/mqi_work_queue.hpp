#ifndef MQI_WORK_QUEUE_HPP
#define MQI_WORK_QUEUE_HPP

#include "mqi_common.hpp"
#include "mqi_error_check.hpp"
#include <atomic>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#include <device_atomic_functions.h>
#endif

namespace mqi {

///< Work item structure for particle transport work distribution
struct work_item_t {
  uint32_t particle_id;   ///< Unique particle identifier
  uint32_t start_history; ///< Starting history number
  uint32_t end_history;   ///< Ending history number
  uint32_t priority;      ///< Work priority (0 = highest)
  uint32_t work_size;     ///< Size of work item (number of particles)
  float weight;           ///< Statistical weight for work item
};

///< Work queue structure with atomic operations for thread-safe access
struct work_queue_t {
  // Common fields for both host and device
  work_item_t *work_items;   ///< Array of work items
  uint32_t total_work_items; ///< Total number of work items
  uint32_t work_item_stride; ///< Stride for work item access

  // Atomic counters for work distribution (using host-side atomics for now)
  std::atomic<uint32_t> next_work_item; ///< Next available work item index
  std::atomic<uint32_t> completed_work; ///< Number of completed work items
  std::atomic<uint32_t> active_workers; ///< Number of currently active workers
  std::atomic<uint32_t> work_steals;    ///< Number of work stealing operations
  std::atomic<uint32_t>
      balance_operations; ///< Number of load balancing operations

  // Configuration parameters
  uint32_t min_work_size;         ///< Minimum work size per item
  uint32_t max_work_size;         ///< Maximum work size per item
  uint32_t work_stealing_enabled; ///< Enable/disable work stealing (0/1)
  uint32_t dynamic_rebalance;     ///< Enable dynamic rebalancing (0/1)
  float load_balance_threshold;   ///< Threshold for load balancing (0.0-1.0)
};

///< Initialize work queue with specified parameters
CUDA_HOST
void initialize_work_queue(work_queue_t *queue, work_item_t *work_items,
                           const uint32_t total_work_items,
                           const uint32_t min_work_size = 1,
                           const uint32_t max_work_size = 1000,
                           const uint32_t work_stealing_enabled = 1,
                           const uint32_t dynamic_rebalance = 1,
                           const float load_balance_threshold = 0.1f);

///< Get next work item atomically (thread-safe)
CUDA_HOST
work_item_t get_next_work_item(work_queue_t *queue);

///< Try to steal work from other threads (work stealing algorithm)
CUDA_HOST
work_item_t steal_work(work_queue_t *queue);

///< Mark work item as completed
CUDA_HOST
void mark_work_completed(work_queue_t *queue, uint32_t work_item_id);

///< Check if work queue is empty
CUDA_HOST
bool is_work_queue_empty(const work_queue_t *queue);

///< Get current work queue statistics
CUDA_HOST
void get_work_queue_stats(const work_queue_t *queue, uint32_t &completed_work,
                          uint32_t &active_workers, uint32_t &work_steals,
                          uint32_t &balance_ops);

///< Reset work queue for new work batch
CUDA_HOST
void reset_work_queue(work_queue_t *queue);

///< Create work items from particle batch
CUDA_HOST
void create_particle_work_items(work_queue_t *queue,
                                const uint32_t num_particles,
                                const uint32_t histories_per_particle = 100,
                                const float base_weight = 1.0f);

///< Load balancing function - redistributes work among threads
CUDA_HOST
void balance_work_load(work_queue_t *queue);

/// Utility functions for work queue management
CUDA_HOST
uint32_t calculate_optimal_work_size(const uint32_t total_work,
                                     const uint32_t num_threads,
                                     const uint32_t min_size = 1,
                                     const uint32_t max_size = 1000);

CUDA_HOST
float calculate_load_balance_efficiency(const work_queue_t *queue);

CUDA_HOST
uint32_t get_thread_local_work_item_id(const work_queue_t *queue);

CUDA_HOST
void update_thread_work_statistics(work_queue_t *queue, uint32_t work_done,
                                   uint32_t time_spent);

} // namespace mqi

#endif // MQI_WORK_QUEUE_HPP
