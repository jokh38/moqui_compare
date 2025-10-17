#ifndef MQI_PHASE1_INTEGRATION_HPP
#define MQI_PHASE1_INTEGRATION_HPP

/// \file
///
/// Phase 1.0 Integration Layer for Moqui Coarse2Fine GPU Optimization
///
/// This module provides a high-level integration layer that orchestrates the
/// persistent thread pool, work queue, and standardized RNG systems implemented
/// in Phase 1.0. It simplifies usage and ensures proper initialization
/// sequencing.
///
/// Key Features:
/// - Unified initialization interface for all Phase 1.0 components
/// - Coordinated configuration management
/// - Integrated performance monitoring
/// - Simplified resource management and cleanup
/// - Backward compatibility with legacy code patterns
/// - Comprehensive error handling and validation

#include "mqi_common.hpp"
#include "mqi_config.hpp"
#include "mqi_error_check.hpp"
#include "mqi_persistent_threads.hpp"
#include "mqi_standardized_rng.hpp"
#include "mqi_threads.hpp"
#include "mqi_work_queue.hpp"

namespace mqi {

// ============================================================================
// PHASE 1.0 INTEGRATION CONFIGURATION
// ============================================================================

///< Unified configuration for Phase 1.0 system
struct phase1_config_t {
  // Thread pool configuration
  thread_pool_config_t thread_pool;

  // Work queue configuration
  uint32_t work_queue_capacity;
  uint32_t target_work_chunk_size;
  float work_stealing_threshold;

  // RNG configuration
  uint64_t master_rng_seed;
  uint32_t rng_batch_size;
  bool rng_standardization_enabled;

  // Performance monitoring
  bool performance_monitoring_enabled;
  uint32_t statistics_update_interval_ms;

  // System-wide settings
  bool verbose_logging;
  bool enable_legacy_compatibility;
  uint32_t max_concurrent_kernels;

  // Memory management
  size_t memory_pool_size_mb;
  bool use_unified_memory;
};

///< Phase 1.0 system state and resource management
struct phase1_system_t {
  // Core components
  persistent_thread_pool_t *thread_pool;
  work_queue_t *work_queue;
  rng_state_t *rng_states;

  // Memory management
  void *system_memory_pool;
  size_t allocated_memory;

  // Performance monitoring
  thread_pool_stats_t thread_stats;
  uint64_t system_start_time;
  uint64_t total_work_items_processed;

  // System state
  bool initialized;
  bool shutdown_requested;
  uint32_t active_kernel_count;

  // Configuration
  phase1_config_t config;
};

// ============================================================================
// PHASE 1.0 SYSTEM LIFECYCLE MANAGEMENT
// ============================================================================

/// Initialize Phase 1.0 system with unified configuration
CUDA_HOST
bool initialize_phase1_system(phase1_system_t *system,
                              const phase1_config_t *config);

/// Shutdown Phase 1.0 system gracefully
CUDA_HOST
void shutdown_phase1_system(phase1_system_t *system);

/// Check if Phase 1.0 system is ready for operation
CUDA_HOST
bool is_phase1_system_ready(const phase1_system_t *system);

/// Reset Phase 1.0 system statistics
CUDA_HOST
void reset_phase1_statistics(phase1_system_t *system);

// ============================================================================
// UNIFIED WORK PROCESSING INTERFACE
// ============================================================================

/// Submit work to Phase 1.0 system (unified interface)
CUDA_HOST
bool submit_work_to_phase1_system(phase1_system_t *system,
                                  const work_item_t *work_items,
                                  const uint32_t num_work_items);

/// Process work items using Phase 1.0 persistent threads
CUDA_HOST
uint32_t
process_work_with_phase1_system(phase1_system_t *system,
                                const uint32_t max_work_items = UINT32_MAX);

/// Synchronize with Phase 1.0 system
CUDA_HOST
bool synchronize_phase1_system(phase1_system_t *system,
                               const uint32_t timeout_ms = 5000);

// ============================================================================
// PERFORMANCE MONITORING AND STATISTICS
// ============================================================================

/// Get comprehensive Phase 1.0 system statistics
CUDA_HOST
void get_phase1_system_statistics(const phase1_system_t *system,
                                  thread_pool_stats_t *thread_stats,
                                  uint32_t *completed_work_items,
                                  uint64_t *system_uptime_us,
                                  float *average_throughput);

/// Get system performance metrics
CUDA_HOST
void get_phase1_performance_metrics(const phase1_system_t *system,
                                    float *gpu_utilization,
                                    float *thread_efficiency,
                                    uint64_t *kernel_launch_overhead_us,
                                    float *memory_bandwidth_utilization);

/// Update system statistics (called internally)
CUDA_HOST
void update_phase1_system_statistics(phase1_system_t *system);

// ============================================================================
// CONFIGURATION MANAGEMENT
// ============================================================================

/// Get default Phase 1.0 configuration
CUDA_HOST
phase1_config_t get_default_phase1_config();

/// Validate Phase 1.0 configuration parameters
CUDA_HOST
bool validate_phase1_config(const phase1_config_t *config);

/// Update Phase 1.0 system configuration dynamically
CUDA_HOST
bool update_phase1_config(phase1_system_t *system,
                          const phase1_config_t *new_config);

/// Get current Phase 1.0 system configuration
CUDA_HOST
void get_phase1_config(const phase1_system_t *system, phase1_config_t *config);

// ============================================================================
// LEGACY COMPATIBILITY INTERFACE
// ============================================================================

/// Legacy compatibility: initialize threads using Phase 1.0 system
CUDA_GLOBAL
void initialize_phase1_threads(mqi::thrd_t *thrds, const uint32_t n_threads,
                               phase1_system_t *system,
                               unsigned long master_seed = 0,
                               unsigned long offset = 0);

/// Legacy compatibility: get work item for thread
CUDA_HOST_DEVICE
inline work_item_t get_work_from_phase1_system(phase1_system_t *system,
                                               const uint32_t thread_id);

/// Legacy compatibility: mark work completed
CUDA_HOST
void mark_work_completed_in_phase1_system(phase1_system_t *system,
                                          const uint32_t work_item_id);

// ============================================================================
// ADVANCED FEATURES AND UTILITIES
// ============================================================================

/// Batch work submission for improved performance
CUDA_HOST
bool submit_work_batch_to_phase1_system(phase1_system_t *system,
                                        const work_item_t *work_items,
                                        const uint32_t num_work_items,
                                        const uint32_t batch_priority = 0);

/// Set up work stealing between thread pools
CUDA_HOST
bool configure_work_stealing(phase1_system_t *system,
                             const bool enable_work_stealing,
                             const uint32_t stealing_threshold = 10);

/// Dynamic thread pool scaling based on workload
CUDA_HOST
bool enable_dynamic_thread_scaling(phase1_system_t *system,
                                   const uint32_t min_threads,
                                   const uint32_t max_threads,
                                   const float scale_up_threshold = 0.8f,
                                   const float scale_down_threshold = 0.3f);

/// Memory pool management for improved allocation performance
CUDA_HOST
bool configure_memory_pool(phase1_system_t *system, const size_t pool_size_mb,
                           const bool use_unified_memory = true);

// ============================================================================
// ERROR HANDLING AND DIAGNOSTICS
// ============================================================================

/// Get system health status
CUDA_HOST
bool get_phase1_system_health(const phase1_system_t *system,
                              char *status_message,
                              const size_t message_buffer_size);

/// Diagnose performance issues
CUDA_HOST
void diagnose_phase1_performance(const phase1_system_t *system,
                                 char *diagnostic_report,
                                 const size_t report_buffer_size);

/// Validate system integrity
CUDA_HOST
bool validate_phase1_system_integrity(const phase1_system_t *system,
                                      char *validation_report,
                                      const size_t report_buffer_size);

// ============================================================================
// INLINE IMPLEMENTATIONS (DEVICE-SIDE FUNCTIONS)
// ============================================================================

#if defined(__CUDACC__)

/// Device function to get work from Phase 1.0 system
CUDA_DEVICE
inline work_item_t get_work_from_phase1_system(phase1_system_t *system,
                                               const uint32_t thread_id) {
  if (!system || !system->initialized || !system->work_queue) {
    return {UINT32_MAX, 0, 0, 0, 0.0f}; // Invalid work item
  }

  return mqi::get_work_for_thread(system->thread_pool, thread_id);
}

#endif // __CUDACC__

// ============================================================================
// DEFAULT CONFIGURATIONS AND CONSTANTS
// ============================================================================

namespace phase1_defaults {
constexpr uint32_t WORK_QUEUE_CAPACITY = mqi::performance::WORK_QUEUE_CAPACITY;
constexpr uint32_t TARGET_WORK_CHUNK_SIZE = 100;
constexpr float WORK_STEALING_THRESHOLD = 0.1f;
constexpr uint64_t MASTER_RNG_SEED = 12345ULL;
constexpr uint32_t RNG_BATCH_SIZE = 1024;
constexpr bool RNG_STANDARDIZATION_ENABLED = true;
constexpr bool PERFORMANCE_MONITORING_ENABLED = true;
constexpr uint32_t STATISTICS_UPDATE_INTERVAL_MS = 1000;
constexpr bool VERBOSE_LOGGING = false;
constexpr bool ENABLE_LEGACY_COMPATIBILITY = true;
constexpr uint32_t MAX_CONCURRENT_KERNELS = 4;
constexpr size_t MEMORY_POOL_SIZE_MB = 256;
constexpr bool USE_UNIFIED_MEMORY = false;

/// Get default Phase 1.0 configuration
inline phase1_config_t get_default_config() {
  return {mqi::defaults::get_default_config(), // thread_pool config
          WORK_QUEUE_CAPACITY,
          TARGET_WORK_CHUNK_SIZE,
          WORK_STEALING_THRESHOLD,
          MASTER_RNG_SEED,
          RNG_BATCH_SIZE,
          RNG_STANDARDIZATION_ENABLED,
          PERFORMANCE_MONITORING_ENABLED,
          STATISTICS_UPDATE_INTERVAL_MS,
          VERBOSE_LOGGING,
          ENABLE_LEGACY_COMPATIBILITY,
          MAX_CONCURRENT_KERNELS,
          MEMORY_POOL_SIZE_MB,
          USE_UNIFIED_MEMORY};
}
} // namespace phase1_defaults

// ============================================================================
// UTILITY MACROS AND HELPERS
// ============================================================================

/// Helper macro for Phase 1.0 system initialization
#define MQI_INITIALIZE_PHASE1_SYSTEM(system_ptr, config_ptr)                   \
  mqi::initialize_phase1_system((system_ptr), (config_ptr))

/// Helper macro for Phase 1.0 system shutdown
#define MQI_SHUTDOWN_PHASE1_SYSTEM(system_ptr)                                 \
  mqi::shutdown_phase1_system((system_ptr))

/// Helper macro for checking Phase 1.0 system readiness
#define MQI_PHASE1_SYSTEM_READY(system_ptr)                                    \
  mqi::is_phase1_system_ready((system_ptr))

/// Helper macro for work submission
#define MQI_SUBMIT_WORK_PHASE1(system_ptr, work_items, count)                  \
  mqi::submit_work_to_phase1_system((system_ptr), (work_items), (count))

} // namespace mqi

#endif // MQI_PHASE1_INTEGRATION_HPP
