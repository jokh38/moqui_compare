#ifndef MQI_PERSISTENT_THREADS_HPP
#define MQI_PERSISTENT_THREADS_HPP

/// \file
///
/// Persistent Thread Pool Management for GPU Monte Carlo Particle Transport
///
/// This module implements the persistent kernel architecture for Phase 1.0 of
/// the Moqui Coarse2Fine GPU optimization project. It replaces the static work
/// distribution with a dynamic, load-balanced architecture using a persistent
/// thread pool pattern.
///
/// Key Features:
/// - Persistent CUDA threads with dedicated streams
/// - Dynamic work distribution via atomic operations
/// - Integration with high-performance work queue (Step 1.1)
/// - Standardized RNG using curandStatePhilox4_32_10_t
/// - Performance monitoring and statistics
/// - Thread-safe operations with minimal overhead

#include "mqi_common.hpp"
#include "mqi_config.hpp"
#include "mqi_error_check.hpp"
#include "mqi_work_queue.hpp"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <random>
#include <thread>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_atomic_functions.h>
#endif

namespace mqi {

///< Performance statistics for persistent thread pool (cache-line aligned)
struct alignas(64) thread_pool_stats_t {
  std::atomic<uint64_t> total_work_items_processed;
  std::atomic<uint64_t>
      total_kernel_time_us; ///< Total kernel execution time in microseconds
  std::atomic<uint64_t>
      total_idle_time_us;               ///< Total idle time waiting for work
  std::atomic<uint32_t> active_threads; ///< Currently active threads
  std::atomic<uint32_t> peak_active_threads; ///< Peak number of active threads
  std::atomic<uint64_t>
      work_stealing_operations;                 ///< Number of work stealing ops
  std::atomic<float> average_work_item_time_us; ///< Average time per work item
  std::atomic<float> gpu_utilization_percent;   ///< Estimated GPU utilization

  // Padding to complete cache line
  char padding[64 - (5 * sizeof(std::atomic<uint64_t>) +
                     3 * sizeof(std::atomic<float>) +
                     2 * sizeof(std::atomic<uint32_t>)) %
                        64];
};

///< Configuration parameters for persistent thread pool
struct thread_pool_config_t {
  uint32_t min_threads;       ///< Minimum number of persistent threads
  uint32_t max_threads;       ///< Maximum number of persistent threads
  uint32_t threads_per_block; ///< Threads per CUDA block (typically 128-256)
  uint32_t target_work_chunk_size; ///< Target work items per thread
  uint32_t work_stealing_enabled;  ///< Enable work stealing (0/1)
  uint32_t dynamic_thread_scaling; ///< Enable dynamic thread scaling (0/1)
  float idle_timeout_ms;           ///< Idle timeout before thread scaling down
  uint32_t performance_monitoring; ///< Enable performance monitoring (0/1)
  uint32_t rng_standardization;    ///< Use standardized RNG (0/1)
};

///< Persistent thread state for each worker thread
struct persistent_thread_state_t {
  uint32_t thread_id;                ///< Unique thread identifier
  uint32_t block_id;                 ///< CUDA block identifier
  uint32_t lane_id;                  ///< Thread lane within block
  bool is_active;                    ///< Thread currently processing work
  bool is_initialized;               ///< Thread has been initialized
  uint32_t work_items_processed;     ///< Number of work items completed
  uint64_t kernel_start_time;        ///< Kernel start timestamp
  uint64_t total_processing_time_us; ///< Total processing time for this thread
  uint32_t work_stealing_attempts;   ///< Number of work stealing attempts
  uint32_t successful_steals;        ///< Successful work steals

  // RNG state - standardized to curandStatePhilox4_32_10_t
#if defined(__CUDACC__)
  curandStatePhilox4_32_10_t
      rng_state; ///< Standardized random number generator
#else
  std::mt19937_64 rng_state; ///< Host fallback RNG
#endif

  // CUDA stream management (host-managed)
#if defined(__CUDACC__)
  int32_t stream_id;        ///< Stream identifier (managed by host)
  uint32_t stream_priority; ///< Stream priority for scheduling
#else
  void *host_stream_handle; ///< Host fallback stream handle
#endif
};

///< Main persistent thread pool structure
struct persistent_thread_pool_t {
  // Thread management
  persistent_thread_state_t *thread_states; ///< Array of thread states
  uint32_t num_threads;                     ///< Total number of threads
  uint32_t num_blocks;                      ///< Number of CUDA blocks
  uint32_t threads_per_block;               ///< Threads per block

  // Work distribution integration
  work_queue_t *work_queue;             ///< Integrated work queue
  std::atomic<bool> shutdown_requested; ///< Graceful shutdown flag
  std::atomic<uint32_t> active_threads; ///< Currently active threads

  // Performance monitoring
  thread_pool_stats_t stats;   ///< Performance statistics
  thread_pool_config_t config; ///< Configuration parameters

  // Synchronization primitives
  std::mutex config_mutex;                ///< Mutex for configuration changes
  std::condition_variable work_available; ///< CV for work availability
  std::atomic<bool> pool_initialized;     ///< Pool initialization status

  // Memory management
  void *device_memory_pool;               ///< Persistent device memory
  size_t memory_pool_size;                ///< Size of memory pool
  std::atomic<uint64_t> allocated_memory; ///< Currently allocated memory
};

// ============================================================================
// CORE THREAD POOL MANAGEMENT FUNCTIONS
// ============================================================================

///< Initialize persistent thread pool with specified configuration
CUDA_HOST
bool initialize_persistent_thread_pool(persistent_thread_pool_t *pool,
                                       work_queue_t *work_queue,
                                       const thread_pool_config_t *config,
                                       const uint64_t master_seed = 12345ULL);

///< Shutdown persistent thread pool gracefully
CUDA_HOST
void shutdown_persistent_thread_pool(persistent_thread_pool_t *pool);

///< Check if thread pool is initialized and ready
CUDA_HOST
bool is_thread_pool_ready(const persistent_thread_pool_t *pool);

// ============================================================================
// WORK DISTRIBUTION AND KERNEL MANAGEMENT
// ============================================================================

///< Launch persistent kernel that continuously processes work items
CUDA_HOST
bool launch_persistent_kernel(persistent_thread_pool_t *pool,
                              const uint32_t grid_size,
                              const uint32_t block_size);

///< Synchronize with all persistent threads
CUDA_HOST
bool synchronize_persistent_threads(persistent_thread_pool_t *pool);

///< Get next work item for a specific thread (with work stealing)
CUDA_HOST
work_item_t get_work_for_thread(persistent_thread_pool_t *pool,
                                const uint32_t thread_id);

// ============================================================================
// PERFORMANCE MONITORING AND STATISTICS
// ============================================================================

///< Get current performance statistics from thread pool
CUDA_HOST
void get_thread_pool_statistics(const persistent_thread_pool_t *pool,
                                thread_pool_stats_t *stats);

///< Reset performance statistics
CUDA_HOST
void reset_thread_pool_statistics(persistent_thread_pool_t *pool);

///< Update thread performance metrics (called by persistent kernel)
CUDA_HOST
void update_thread_performance(persistent_thread_pool_t *pool,
                               const uint32_t thread_id,
                               const uint64_t work_time_us);

// ============================================================================
// CONFIGURATION AND DYNAMIC MANAGEMENT
// ============================================================================

///< Update thread pool configuration dynamically
CUDA_HOST
bool update_thread_pool_config(persistent_thread_pool_t *pool,
                               const thread_pool_config_t *new_config);

///< Get current thread pool configuration
CUDA_HOST
void get_thread_pool_config(const persistent_thread_pool_t *pool,
                            thread_pool_config_t *config);

///< Scale number of active threads based on workload
CUDA_HOST
bool scale_thread_pool(persistent_thread_pool_t *pool,
                       const uint32_t target_threads);

// ============================================================================
// UTILITY AND HELPER FUNCTIONS
// ============================================================================

///< Calculate optimal thread count for given GPU and workload
CUDA_HOST
uint32_t calculate_optimal_thread_count(const uint32_t sm_count,
                                        const uint32_t max_threads_per_sm,
                                        const uint32_t workload_size);

///< Validate thread pool configuration parameters
CUDA_HOST
bool validate_thread_pool_config(const thread_pool_config_t *config);

///< Get recommended configuration for specific GPU architecture
CUDA_HOST
void get_recommended_config(const int cuda_device,
                            thread_pool_config_t *config);

// ============================================================================
// DEVICE-SIDE FUNCTIONS (accessed by persistent kernel)
// ============================================================================

#if defined(__CUDACC__)

/// Device function to get thread-local state
CUDA_DEVICE
persistent_thread_state_t *get_thread_state(persistent_thread_pool_t *pool);

/// Device function to initialize thread RNG state
CUDA_DEVICE
void initialize_thread_rng(persistent_thread_state_t *thread_state,
                           const uint64_t master_seed);

/// Device function to process a single work item
CUDA_DEVICE
bool process_work_item(persistent_thread_state_t *thread_state,
                       work_item_t work_item, work_queue_t *work_queue);

/// Device function to report work completion
CUDA_DEVICE
void report_work_completion(persistent_thread_pool_t *pool,
                            persistent_thread_state_t *thread_state,
                            const work_item_t completed_work);

/// Device function to check for shutdown signal
CUDA_DEVICE
bool should_shutdown(const persistent_thread_pool_t *pool);

#endif // __CUDACC__

// ============================================================================
// DEFAULT CONFIGURATIONS AND CONSTANTS
// ============================================================================

namespace defaults {
constexpr uint32_t MIN_THREADS = mqi::threads::MIN_THREADS;
constexpr uint32_t MAX_THREADS = mqi::threads::MAX_THREADS;
constexpr uint32_t THREADS_PER_BLOCK = mqi::threads::THREADS_PER_BLOCK;
constexpr uint32_t TARGET_WORK_CHUNK_SIZE =
    mqi::performance::TARGET_WORK_CHUNK_SIZE;
constexpr bool WORK_STEALING_ENABLED = mqi::performance::WORK_STEALING_ENABLED;
constexpr bool DYNAMIC_THREAD_SCALING =
    mqi::performance::DYNAMIC_THREAD_SCALING;
constexpr float IDLE_TIMEOUT_MS = mqi::threads::IDLE_TIMEOUT_MS;
constexpr bool PERFORMANCE_MONITORING =
    mqi::performance::PERFORMANCE_MONITORING;
constexpr bool RNG_STANDARDIZATION = mqi::performance::RNG_STANDARDIZATION;

///< Get default configuration for thread pool
inline thread_pool_config_t get_default_config() {
  return {MIN_THREADS,           MAX_THREADS,
          THREADS_PER_BLOCK,     TARGET_WORK_CHUNK_SIZE,
          WORK_STEALING_ENABLED, DYNAMIC_THREAD_SCALING,
          IDLE_TIMEOUT_MS,       PERFORMANCE_MONITORING,
          RNG_STANDARDIZATION};
}
} // namespace defaults

} // namespace mqi

#endif // MQI_PERSISTENT_THREADS_HPP
