#include "mqi_persistent_threads.hpp"
#include "mqi_error_check.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>

// CUDA runtime includes for host-side CUDA functions
#if defined(__CUDACC__)
#include <cuda_runtime.h>
#include <device_atomic_functions.h>
#endif

namespace mqi {

// ============================================================================
// CUDA KERNEL: Persistent Worker Kernel
// ============================================================================

#if defined(__CUDACC__)

/// Persistent kernel that continuously processes work items
template <typename T>
__global__ void persistent_worker_kernel(
    persistent_thread_pool_t *pool, work_queue_t *work_queue,
    mqi::node_t<T> *world, mqi::vertex_t<T> *vertices,
    uint32_t *tracked_particles, uint32_t *scorer_offset_vector = nullptr,
    bool score_local_deposit = true) {
  // Get thread-local state
  const uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  persistent_thread_state_t *thread_state = &pool->thread_states[thread_id];

  // Initialize thread state
  thread_state->thread_id = thread_id;
  thread_state->block_id = blockIdx.x;
  thread_state->lane_id = threadIdx.x;
  thread_state->is_active = false;
  thread_state->is_initialized = true;
  thread_state->work_items_processed = 0;
  thread_state->work_stealing_attempts = 0;
  thread_state->successful_steals = 0;
  thread_state->total_processing_time_us = 0;

  // Initialize RNG with standardized state
  initialize_thread_rng(thread_state, 12345ULL + thread_id);

  // Note: CUDA streams are managed by host, not created in device code
  thread_state->stream_id = static_cast<int32_t>(thread_id);
  thread_state->stream_priority = 0;

  // Atomic increment to mark thread as ready
  atomicAdd(&pool->active_threads, 1);

  // Main persistent work loop
  while (!should_shutdown(pool)) {
    // Get work item from queue
    work_item_t work_item = get_work_for_thread(pool, thread_id);

    if (work_item.particle_id != UINT32_MAX) {
      // Mark thread as active
      thread_state->is_active = true;
      thread_state->kernel_start_time = clock64();

      // Process the work item
      bool success = process_work_item(thread_state, work_item, work_queue);

      if (success) {
        // Update performance metrics
        uint64_t work_time_us =
            (clock64() - thread_state->kernel_start_time) / 1000;
        thread_state->total_processing_time_us += work_time_us;
        thread_state->work_items_processed++;

        // Report completion
        report_work_completion(pool, thread_state, work_item);
      }

      // Mark thread as inactive
      thread_state->is_active = false;
    } else {
// No work available, brief pause to reduce CPU/GPU contention
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
      __nanosleep(1000); // 1 microsecond pause for Volta and newer
#else
      // Fallback for older architectures
      for (volatile int i = 0; i < 100; ++i) {
      }
#endif
    }
  }

  // Cleanup thread state (streams managed by host)
  thread_state->is_initialized = false;
  atomicSub(&pool->active_threads, 1);
}

#endif // __CUDACC__

// ============================================================================
// HOST-SIDE IMPLEMENTATION
// ============================================================================

CUDA_HOST
bool initialize_persistent_thread_pool(persistent_thread_pool_t *pool,
                                       work_queue_t *work_queue,
                                       const thread_pool_config_t *config,
                                       const uint64_t master_seed) {
  if (!pool || !work_queue || !config) {
    return false;
  }

  // Validate configuration
  if (!validate_thread_pool_config(config)) {
    return false;
  }

  // Initialize pool structure
  std::memset(pool, 0, sizeof(persistent_thread_pool_t));
  pool->work_queue = work_queue;
  pool->config = *config;
  pool->shutdown_requested = false;
  pool->active_threads = 0;
  pool->pool_initialized = false;

  // Calculate optimal thread count
#if defined(__CUDACC__)
  int device_id;
  cudaGetDevice(&device_id);
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, device_id);

  uint32_t optimal_threads = calculate_optimal_thread_count(
      device_prop.multiProcessorCount, device_prop.maxThreadsPerMultiProcessor,
      work_queue->total_work_items);
#else
  // Host-only fallback: use reasonable defaults
  uint32_t optimal_threads =
      calculate_optimal_thread_count(20,   // Assume 20 SMs
                                     2048, // Assume max threads per SM
                                     work_queue->total_work_items);
#endif

  pool->num_threads = std::min(optimal_threads, config->max_threads);
  pool->num_threads = std::max(pool->num_threads, config->min_threads);
  pool->threads_per_block = config->threads_per_block;
  pool->num_blocks = (pool->num_threads + pool->threads_per_block - 1) /
                     pool->threads_per_block;

  // Allocate device memory for thread states
  size_t thread_states_size =
      pool->num_threads * sizeof(persistent_thread_state_t);
#if defined(__CUDACC__)
  cudaError_t error = cudaMalloc(&pool->thread_states, thread_states_size);
  if (error != cudaSuccess) {
    return false;
  }

  error = cudaMemset(pool->thread_states, 0, thread_states_size);
  if (error != cudaSuccess) {
    cudaFree(pool->thread_states);
    pool->thread_states = nullptr;
    return false;
  }

  // Allocate persistent device memory pool
  pool->memory_pool_size = 64 * 1024 * 1024; // 64MB default
  error = cudaMalloc(&pool->device_memory_pool, pool->memory_pool_size);
  if (error != cudaSuccess) {
    cudaFree(pool->thread_states);
    pool->thread_states = nullptr;
    return false;
  }
#else
  // Host-only fallback: allocate using regular malloc
  pool->thread_states =
      static_cast<persistent_thread_state_t *>(std::malloc(thread_states_size));
  if (!pool->thread_states) {
    return false;
  }
  std::memset(pool->thread_states, 0, thread_states_size);

  // Allocate persistent memory pool
  pool->memory_pool_size = 64 * 1024 * 1024; // 64MB default
  pool->device_memory_pool = std::malloc(pool->memory_pool_size);
  if (!pool->device_memory_pool) {
    std::free(pool->thread_states);
    pool->thread_states = nullptr;
    return false;
  }
#endif
  pool->allocated_memory = 0;

  // Initialize statistics
  std::memset(&pool->stats, 0, sizeof(thread_pool_stats_t));

  // Set initialization flag
  pool->pool_initialized = true;

  return true;
}

CUDA_HOST
void shutdown_persistent_thread_pool(persistent_thread_pool_t *pool) {
  if (!pool || !pool->pool_initialized) {
    return;
  }

  // Signal shutdown
  pool->shutdown_requested = true;

  // Wait for all threads to complete
  if (pool->num_threads > 0) {
    synchronize_persistent_threads(pool);
  }

  // Free device memory
  if (pool->thread_states) {
#if defined(__CUDACC__)
    cudaFree(pool->thread_states);
#else
    std::free(pool->thread_states);
#endif
    pool->thread_states = nullptr;
  }

  if (pool->device_memory_pool) {
#if defined(__CUDACC__)
    cudaFree(pool->device_memory_pool);
#else
    std::free(pool->device_memory_pool);
#endif
    pool->device_memory_pool = nullptr;
  }

  // Mark pool as uninitialized
  pool->pool_initialized = false;
}

CUDA_HOST
bool is_thread_pool_ready(const persistent_thread_pool_t *pool) {
  return pool && pool->pool_initialized && pool->thread_states;
}

CUDA_HOST
bool launch_persistent_kernel(persistent_thread_pool_t *pool,
                              const uint32_t grid_size,
                              const uint32_t block_size) {
  if (!pool || !pool->pool_initialized) {
    return false;
  }

  // Note: This is a placeholder for the actual kernel launch
  // The real implementation would need to be templated for the specific
  // physics type (float/double) and world geometry

#if defined(__CUDACC__)
  // This would be the actual kernel launch:
  // persistent_worker_kernel<<<grid_size, block_size>>>(
  //     pool->device_pool,
  //     pool->work_queue,
  //     world,
  //     vertices,
  //     tracked_particles,
  //     scorer_offset_vector,
  //     score_local_deposit
  // );

  // Check for launch errors
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return false;
  }
#endif

  return true;
}

CUDA_HOST
bool synchronize_persistent_threads(persistent_thread_pool_t *pool) {
  if (!pool) {
    return false;
  }

#if defined(__CUDACC__)
  cudaError_t error = cudaDeviceSynchronize();
  return (error == cudaSuccess);
#else
  // Host-only fallback: nothing to synchronize
  return true;
#endif
}

CUDA_HOST
work_item_t get_work_for_thread(persistent_thread_pool_t *pool,
                                const uint32_t thread_id) {
  if (!pool || !pool->work_queue) {
    work_item_t empty_item = {UINT32_MAX, 0, 0, 0, 0, 0.0f};
    return empty_item;
  }

  // Try to get work from primary queue
  work_item_t work_item = get_next_work_item(pool->work_queue);

  if (work_item.particle_id != UINT32_MAX) {
    return work_item;
  }

  // If no work available, try work stealing if enabled
  if (pool->config.work_stealing_enabled) {
    work_item = steal_work(pool->work_queue);
    if (work_item.particle_id != UINT32_MAX) {
#if defined(__CUDACC__)
      atomicAdd(&pool->stats.work_stealing_operations, 1);
#else
      pool->stats.work_stealing_operations.fetch_add(1);
#endif
    }
  }

  return work_item;
}

CUDA_HOST
void get_thread_pool_statistics(const persistent_thread_pool_t *pool,
                                thread_pool_stats_t *stats) {
  if (!pool || !stats) {
    return;
  }

  // Copy atomic values
  stats->total_work_items_processed =
      pool->stats.total_work_items_processed.load();
  stats->total_kernel_time_us = pool->stats.total_kernel_time_us.load();
  stats->total_idle_time_us = pool->stats.total_idle_time_us.load();
  stats->active_threads = pool->active_threads.load();
  stats->peak_active_threads = pool->stats.peak_active_threads.load();
  stats->work_stealing_operations = pool->stats.work_stealing_operations.load();
  stats->average_work_item_time_us =
      pool->stats.average_work_item_time_us.load();
  stats->gpu_utilization_percent = pool->stats.gpu_utilization_percent.load();
}

CUDA_HOST
void reset_thread_pool_statistics(persistent_thread_pool_t *pool) {
  if (!pool) {
    return;
  }

  pool->stats.total_work_items_processed = 0;
  pool->stats.total_kernel_time_us = 0;
  pool->stats.total_idle_time_us = 0;
  pool->stats.active_threads = 0;
  pool->stats.peak_active_threads = 0;
  pool->stats.work_stealing_operations = 0;
  pool->stats.average_work_item_time_us = 0.0f;
  pool->stats.gpu_utilization_percent = 0.0f;
}

CUDA_HOST
bool update_thread_pool_config(persistent_thread_pool_t *pool,
                               const thread_pool_config_t *new_config) {
  if (!pool || !new_config || !pool->pool_initialized) {
    return false;
  }

  // Validate new configuration
  if (!validate_thread_pool_config(new_config)) {
    return false;
  }

  std::lock_guard<std::mutex> lock(pool->config_mutex);

  // Update configuration
  pool->config = *new_config;

  // If thread count changed significantly, consider scaling
  if (pool->config.dynamic_thread_scaling) {
    uint32_t target_threads =
        std::min(new_config->max_threads, new_config->min_threads);
    if (abs((int)target_threads - (int)pool->num_threads) >
        (int)(pool->num_threads / 4)) {
      scale_thread_pool(pool, target_threads);
    }
  }

  return true;
}

CUDA_HOST
void get_thread_pool_config(const persistent_thread_pool_t *pool,
                            thread_pool_config_t *config) {
  if (!pool || !config) {
    return;
  }

  std::lock_guard<std::mutex> lock(
      const_cast<persistent_thread_pool_t *>(pool)->config_mutex);
  *config = pool->config;
}

CUDA_HOST
bool scale_thread_pool(persistent_thread_pool_t *pool,
                       const uint32_t target_threads) {
  if (!pool || !pool->pool_initialized) {
    return false;
  }

  // For now, scaling requires full shutdown and reinitialization
  // In a more sophisticated implementation, this could be done dynamically
  uint32_t old_threads = pool->num_threads;
  uint32_t new_threads = std::max(target_threads, pool->config.min_threads);
  new_threads = std::min(new_threads, pool->config.max_threads);

  if (new_threads == old_threads) {
    return true; // No scaling needed
  }

  // This would require more complex implementation for dynamic scaling
  // For now, return false to indicate scaling not implemented
  return false;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

CUDA_HOST
uint32_t calculate_optimal_thread_count(const uint32_t sm_count,
                                        const uint32_t max_threads_per_sm,
                                        const uint32_t workload_size) {
  // Calculate theoretical maximum threads
  uint32_t max_threads = sm_count * max_threads_per_sm;

  // Adjust for workload size
  uint32_t workload_threads = std::min(workload_size, max_threads);

  // Target 80% occupancy for persistent kernels
  uint32_t target_threads = static_cast<uint32_t>(workload_threads * 0.8f);

  // Ensure reasonable bounds
  target_threads = std::max(target_threads, defaults::MIN_THREADS);
  target_threads = std::min(target_threads, defaults::MAX_THREADS);

  return target_threads;
}

CUDA_HOST
bool validate_thread_pool_config(const thread_pool_config_t *config) {
  if (!config) {
    return false;
  }

  // Check thread counts
  if (config->min_threads == 0 || config->max_threads == 0) {
    return false;
  }

  if (config->min_threads > config->max_threads) {
    return false;
  }

  // Check threads per block
  if (config->threads_per_block == 0 || config->threads_per_block > 1024) {
    return false;
  }

  // Check work chunk size
  if (config->target_work_chunk_size == 0) {
    return false;
  }

  // Check idle timeout
  if (config->idle_timeout_ms < 0.0f) {
    return false;
  }

  return true;
}

CUDA_HOST
void get_recommended_config(const int cuda_device,
                            thread_pool_config_t *config) {
  if (!config) {
    return;
  }

  // Start with defaults
  *config = defaults::get_default_config();

#if defined(__CUDACC__)
  // Get device properties
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, cuda_device);

  // Adjust based on device capabilities
  config->max_threads =
      device_prop.maxThreadsPerMultiProcessor * device_prop.multiProcessorCount;
  config->max_threads = std::min(config->max_threads, defaults::MAX_THREADS);

  // Optimal threads per block based on architecture
  if (device_prop.major >= 7) {
    // Volta and newer: 128 threads per block is optimal for persistent kernels
    config->threads_per_block = 128;
  } else if (device_prop.major >= 6) {
    // Pascal: 256 threads per block
    config->threads_per_block = 256;
  } else {
    // Older architectures: 512 threads per block
    config->threads_per_block = 512;
  }

  // Adjust minimum threads based on SM count
  config->min_threads =
      std::max(defaults::MIN_THREADS, device_prop.multiProcessorCount);
#else
  // Host-only fallback: use reasonable defaults
  config->max_threads = defaults::MAX_THREADS;
  config->threads_per_block = 128; // Good default for modern GPUs
  config->min_threads = defaults::MIN_THREADS;
#endif
}

// ============================================================================
// DEVICE-SIDE FUNCTIONS
// ============================================================================

#if defined(__CUDACC__)

CUDA_DEVICE
persistent_thread_state_t *get_thread_state(persistent_thread_pool_t *pool) {
  const uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  return &pool->thread_states[thread_id];
}

CUDA_DEVICE
void initialize_thread_rng(persistent_thread_state_t *thread_state,
                           const uint64_t master_seed) {
  // Initialize standardized RNG: curandStatePhilox4_32_10_t
  curand_init(master_seed + thread_state->thread_id, thread_state->block_id,
              thread_state->lane_id, &thread_state->rng_state);
}

CUDA_DEVICE
bool process_work_item(persistent_thread_state_t *thread_state,
                       work_item_t work_item, work_queue_t *work_queue) {
  // This is a placeholder for the actual work processing
  // In the real implementation, this would call the physics simulation
  // for the specified particles/histories

  // Simulate work processing time (this would be actual physics simulation)
  uint64_t start_time = clock64();

  // TODO: Replace with actual particle transport simulation
  // This would integrate with the existing transport_particles_patient kernel
  // but adapted for the persistent kernel pattern

  uint64_t end_time = clock64();
  thread_state->total_processing_time_us += (end_time - start_time) / 1000;

  return true;
}

CUDA_DEVICE
void report_work_completion(persistent_thread_pool_t *pool,
                            persistent_thread_state_t *thread_state,
                            const work_item_t completed_work) {
  // Update global statistics atomically
  atomicAdd(&pool->stats.total_work_items_processed, 1);

  // Mark work as completed in the work queue
  mark_work_completed(pool->work_queue, completed_work.particle_id);
}

CUDA_DEVICE
bool should_shutdown(const persistent_thread_pool_t *pool) {
  return pool->shutdown_requested.load();
}

#endif // __CUDACC__

} // namespace mqi
