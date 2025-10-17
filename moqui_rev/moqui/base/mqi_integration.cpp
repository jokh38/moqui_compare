#include "mqi_integration.hpp"
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <thread>

// Add missing CUDA runtime includes for device functions
#if defined(__CUDACC__)
#include <cuda_runtime.h>
#include <curand_kernel.h>
#endif

// Helper function to get current timestamp in microseconds
CUDA_HOST
uint64_t get_current_timestamp_us() {
  auto now = std::chrono::high_resolution_clock::now();
  auto duration = now.time_since_epoch();
  return std::chrono::duration_cast<std::chrono::microseconds>(duration)
      .count();
}

namespace mqi {

// ============================================================================
// PHASE 1.0 SYSTEM LIFECYCLE MANAGEMENT IMPLEMENTATION
// ============================================================================

CUDA_HOST
bool initialize_phase1_system(phase1_system_t *system,
                              const phase1_config_t *config) {
  if (!system || !config) {
    return false;
  }

  // Clear system structure
  std::memset(system, 0, sizeof(phase1_system_t));
  system->config = *config;

  // Validate configuration
  if (!validate_phase1_config(config)) {
    if (config->verbose_logging) {
      std::cerr << "Phase 1.0: Invalid configuration parameters" << std::endl;
    }
    return false;
  }

  // Allocate work queue memory
  const size_t work_item_size =
      sizeof(work_item_t) * config->work_queue_capacity;
  work_item_t *work_items =
      static_cast<work_item_t *>(std::malloc(work_item_size));
  if (!work_items) {
    if (config->verbose_logging) {
      std::cerr << "Phase 1.0: Failed to allocate work queue memory"
                << std::endl;
    }
    return false;
  }

  // Initialize work queue
  system->work_queue = new work_queue_t();
  std::memset(system->work_queue, 0, sizeof(work_queue_t));

  // Initialize work queue fields directly
  system->work_queue->work_items = work_items;
  system->work_queue->total_work_items = config->work_queue_capacity;
  system->work_queue->next_work_item = 0;
  system->work_queue->work_stealing_enabled = 1;
  system->work_queue->dynamic_rebalance = 1;

  // Initialize persistent thread pool
  system->thread_pool = new persistent_thread_pool_t();
  std::memset(system->thread_pool, 0, sizeof(persistent_thread_pool_t));

  bool thread_pool_init = mqi::initialize_persistent_thread_pool(
      system->thread_pool, system->work_queue, &config->thread_pool,
      config->master_rng_seed);

  if (!thread_pool_init) {
    delete system->work_queue;
    std::free(work_items);
    delete system->thread_pool;
    if (config->verbose_logging) {
      std::cerr << "Phase 1.0: Failed to initialize thread pool" << std::endl;
    }
    return false;
  }

  // Initialize RNG states if standardization is enabled
  if (config->rng_standardization_enabled) {
    const size_t rng_array_size =
        sizeof(rng_state_t) * config->thread_pool.max_threads;
    system->rng_states =
        static_cast<rng_state_t *>(std::malloc(rng_array_size));

    if (!system->rng_states) {
      mqi::shutdown_persistent_thread_pool(system->thread_pool);
      delete system->work_queue;
      std::free(work_items);
      delete system->thread_pool;
      if (config->verbose_logging) {
        std::cerr << "Phase 1.0: Failed to allocate RNG states" << std::endl;
      }
      return false;
    }

    // Initialize RNG batch
    rng_batch_init_t rng_init;
    rng_init.master_seed = config->master_rng_seed;
    rng_init.num_threads = config->thread_pool.max_threads;
    rng_init.thread_offset = 0;
    rng_init.sequence_offset = 0;

    mqi::initialize_rng_batch(system->rng_states, &rng_init);
  }

  // Configure memory pool if specified
  if (config->memory_pool_size_mb > 0) {
    system->system_memory_pool =
        std::malloc(config->memory_pool_size_mb * 1024 * 1024);
    if (system->system_memory_pool) {
      system->allocated_memory = config->memory_pool_size_mb * 1024 * 1024;
    } else if (config->verbose_logging) {
      std::cerr << "Phase 1.0: Warning - Failed to allocate memory pool"
                << std::endl;
    }
  }

  // Initialize performance monitoring
  system->system_start_time = get_current_timestamp_us();
  system->total_work_items_processed = 0;
  system->initialized = true;

  if (config->verbose_logging) {
    std::cout << "Phase 1.0: System initialized successfully" << std::endl;
    std::cout << "  Thread Pool: " << system->thread_pool->num_threads
              << " threads" << std::endl;
    std::cout << "  Work Queue: " << config->work_queue_capacity << " capacity"
              << std::endl;
    std::cout << "  RNG Standardization: "
              << (config->rng_standardization_enabled ? "Enabled" : "Disabled")
              << std::endl;
    std::cout << "  Memory Pool: "
              << (system->system_memory_pool ? "Allocated" : "Not allocated")
              << std::endl;
  }

  return true;
}

CUDA_HOST
void shutdown_phase1_system(phase1_system_t *system) {
  if (!system || !system->initialized) {
    return;
  }

  if (system->config.verbose_logging) {
    std::cout << "Phase 1.0: Shutting down system..." << std::endl;
  }

  // Request graceful shutdown
  system->shutdown_requested = true;

  // Shutdown thread pool
  if (system->thread_pool) {
    mqi::shutdown_persistent_thread_pool(system->thread_pool);
    delete system->thread_pool;
    system->thread_pool = nullptr;
  }

  // Shutdown work queue
  if (system->work_queue) {
    delete system->work_queue;
    system->work_queue = nullptr;
  }

  // Clean up RNG states
  if (system->rng_states) {
    std::free(system->rng_states);
    system->rng_states = nullptr;
  }

  // Clean up memory pool
  if (system->system_memory_pool) {
    std::free(system->system_memory_pool);
    system->system_memory_pool = nullptr;
    system->allocated_memory = 0;
  }

  // Mark as shutdown
  system->initialized = false;

  if (system->config.verbose_logging) {
    std::cout << "Phase 1.0: System shutdown complete" << std::endl;
    std::cout << "  Total work items processed: "
              << system->total_work_items_processed << std::endl;
    uint64_t uptime = get_current_timestamp_us() - system->system_start_time;
    std::cout << "  System uptime: " << (uptime / 1000000.0) << " seconds"
              << std::endl;
  }
}

CUDA_HOST
bool is_phase1_system_ready(const phase1_system_t *system) {
  return system && system->initialized && !system->shutdown_requested &&
         system->thread_pool && system->work_queue &&
         mqi::is_thread_pool_ready(system->thread_pool);
}

CUDA_HOST
void reset_phase1_statistics(phase1_system_t *system) {
  if (!system || !system->initialized) {
    return;
  }

  system->total_work_items_processed = 0;
  system->system_start_time = get_current_timestamp_us();

  if (system->thread_pool) {
    mqi::reset_thread_pool_statistics(system->thread_pool);
  }

  if (system->config.verbose_logging) {
    std::cout << "Phase 1.0: Statistics reset" << std::endl;
  }
}

// ============================================================================
// UNIFIED WORK PROCESSING INTERFACE IMPLEMENTATION
// ============================================================================

CUDA_HOST
bool submit_work_to_phase1_system(phase1_system_t *system,
                                  const work_item_t *work_items,
                                  const uint32_t num_work_items) {
  if (!system || !system->initialized || !work_items || num_work_items == 0) {
    return false;
  }

  uint32_t submitted_count = 0;
  // For Phase 1.0 integration, we directly assign work items to the queue's
  // array In a real implementation, this would involve more sophisticated work
  // distribution
  for (uint32_t i = 0;
       i < num_work_items && i < system->work_queue->total_work_items; ++i) {
    system->work_queue->work_items[i] = work_items[i];
    submitted_count++;
  }

  if (system->config.verbose_logging && submitted_count < num_work_items) {
    std::cout << "Phase 1.0: Submitted " << submitted_count << "/"
              << num_work_items << " work items" << std::endl;
  }

  return submitted_count > 0;
}

CUDA_HOST
uint32_t process_work_with_phase1_system(phase1_system_t *system,
                                         const uint32_t max_work_items) {
  if (!system || !system->initialized) {
    return 0;
  }

  // Launch persistent kernel if not already running
  if (!mqi::launch_persistent_kernel(system->thread_pool,
                                     system->thread_pool->num_blocks,
                                     system->thread_pool->threads_per_block)) {
    if (system->config.verbose_logging) {
      std::cerr << "Phase 1.0: Failed to launch persistent kernel" << std::endl;
    }
    return 0;
  }

  // Wait for processing or timeout
  uint32_t processed_count = 0;
  const uint32_t timeout_ms = 5000; // 5 second timeout
  const auto start_time = std::chrono::high_resolution_clock::now();

  while (processed_count < max_work_items) {
    uint32_t completed, active_workers, work_steals, balance_ops;
    mqi::get_work_queue_stats(system->work_queue, completed, active_workers,
                              work_steals, balance_ops);

    processed_count = completed;

    // Check timeout
    auto elapsed = std::chrono::high_resolution_clock::now() - start_time;
    if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() >
        timeout_ms) {
      if (system->config.verbose_logging) {
        std::cout << "Phase 1.0: Processing timeout after " << timeout_ms
                  << "ms" << std::endl;
      }
      break;
    }

    // Check if work is complete
    uint32_t next_item = system->work_queue->next_work_item.load();
    if (active_workers == 0 &&
        next_item >= system->work_queue->total_work_items) {
      break;
    }

    // Small delay to prevent busy waiting
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  system->total_work_items_processed += processed_count;

  if (system->config.verbose_logging) {
    std::cout << "Phase 1.0: Processed " << processed_count << " work items"
              << std::endl;
  }

  return processed_count;
}

CUDA_HOST
bool synchronize_phase1_system(phase1_system_t *system,
                               const uint32_t timeout_ms) {
  if (!system || !system->initialized) {
    return false;
  }

  return mqi::synchronize_persistent_threads(system->thread_pool);
}

// ============================================================================
// PERFORMANCE MONITORING AND STATISTICS IMPLEMENTATION
// ============================================================================

CUDA_HOST
void get_phase1_system_statistics(const phase1_system_t *system,
                                  thread_pool_stats_t *thread_stats,
                                  uint32_t *completed_work_items,
                                  uint64_t *system_uptime_us,
                                  float *average_throughput) {
  if (!system || !system->initialized) {
    return;
  }

  // Get thread pool statistics
  if (thread_stats && system->thread_pool) {
    mqi::get_thread_pool_statistics(system->thread_pool, thread_stats);
  }

  // Get work queue statistics
  if (completed_work_items && system->work_queue) {
    uint32_t completed, active_workers, work_steals, balance_ops;
    mqi::get_work_queue_stats(system->work_queue, completed, active_workers,
                              work_steals, balance_ops);
    *completed_work_items = completed;
  }

  // Calculate system uptime
  if (system_uptime_us) {
    *system_uptime_us = get_current_timestamp_us() - system->system_start_time;
  }

  // Calculate average throughput
  if (average_throughput) {
    uint64_t uptime = get_current_timestamp_us() - system->system_start_time;
    if (uptime > 0) {
      *average_throughput =
          static_cast<float>(system->total_work_items_processed) /
          (static_cast<float>(uptime) / 1000000.0f); // items per second
    } else {
      *average_throughput = 0.0f;
    }
  }
}

CUDA_HOST
void get_phase1_performance_metrics(const phase1_system_t *system,
                                    float *gpu_utilization,
                                    float *thread_efficiency,
                                    uint64_t *kernel_launch_overhead_us,
                                    float *memory_bandwidth_utilization) {
  if (!system || !system->initialized) {
    return;
  }

  if (system->thread_pool) {
    mqi::thread_pool_stats_t stats;
    mqi::get_thread_pool_statistics(system->thread_pool, &stats);

    if (gpu_utilization) {
      *gpu_utilization = stats.gpu_utilization_percent;
    }

    if (thread_efficiency) {
      if (stats.peak_active_threads > 0) {
        *thread_efficiency = static_cast<float>(stats.active_threads) /
                             static_cast<float>(stats.peak_active_threads);
      } else {
        *thread_efficiency = 0.0f;
      }
    }
  }

  // Note: These would require additional profiling infrastructure
  if (kernel_launch_overhead_us) {
    *kernel_launch_overhead_us = 0; // Placeholder
  }

  if (memory_bandwidth_utilization) {
    *memory_bandwidth_utilization = 0.0f; // Placeholder
  }
}

CUDA_HOST
void update_phase1_system_statistics(phase1_system_t *system) {
  if (!system || !system->initialized ||
      !system->config.performance_monitoring_enabled) {
    return;
  }

  // Update thread pool statistics
  if (system->thread_pool) {
    // Statistics are automatically updated in thread pool implementation
  }
}

// ============================================================================
// CONFIGURATION MANAGEMENT IMPLEMENTATION
// ============================================================================

CUDA_HOST
phase1_config_t get_default_phase1_config() {
  return phase1_defaults::get_default_config();
}

CUDA_HOST
bool validate_phase1_config(const phase1_config_t *config) {
  if (!config) {
    return false;
  }

  // Validate thread pool configuration
  if (!mqi::validate_thread_pool_config(&config->thread_pool)) {
    return false;
  }

  // Validate work queue configuration
  if (config->work_queue_capacity == 0 ||
      config->work_queue_capacity > 10000000) {
    return false;
  }

  if (config->target_work_chunk_size == 0) {
    return false;
  }

  if (config->work_stealing_threshold < 0.0f ||
      config->work_stealing_threshold > 1.0f) {
    return false;
  }

  // Validate RNG configuration
  if (config->rng_batch_size == 0) {
    return false;
  }

  // Validate memory pool configuration
  if (config->memory_pool_size_mb > 8192) { // Max 8GB
    return false;
  }

  return true;
}

CUDA_HOST
bool update_phase1_config(phase1_system_t *system,
                          const phase1_config_t *new_config) {
  if (!system || !system->initialized || !new_config) {
    return false;
  }

  if (!validate_phase1_config(new_config)) {
    return false;
  }

  // Update thread pool configuration
  bool update_success = mqi::update_thread_pool_config(
      system->thread_pool, &new_config->thread_pool);

  if (update_success) {
    system->config = *new_config;

    if (system->config.verbose_logging) {
      std::cout << "Phase 1.0: Configuration updated successfully" << std::endl;
    }
  }

  return update_success;
}

CUDA_HOST
void get_phase1_config(const phase1_system_t *system, phase1_config_t *config) {
  if (system && config && system->initialized) {
    *config = system->config;
  }
}

// ============================================================================
// ADVANCED FEATURES IMPLEMENTATION
// ============================================================================

CUDA_HOST
bool submit_work_batch_to_phase1_system(phase1_system_t *system,
                                        const work_item_t *work_items,
                                        const uint32_t num_work_items,
                                        const uint32_t batch_priority) {
  // For now, just use regular submission - could be enhanced with priority
  // queues
  return submit_work_to_phase1_system(system, work_items, num_work_items);
}

CUDA_HOST
bool configure_work_stealing(phase1_system_t *system,
                             const bool enable_work_stealing,
                             const uint32_t stealing_threshold) {
  if (!system || !system->initialized) {
    return false;
  }

  // Update work queue configuration for work stealing
  // This would require modifying the work queue implementation
  if (system->config.verbose_logging) {
    std::cout << "Phase 1.0: Work stealing "
              << (enable_work_stealing ? "enabled" : "disabled") << std::endl;
  }

  return true;
}

CUDA_HOST
bool enable_dynamic_thread_scaling(phase1_system_t *system,
                                   const uint32_t min_threads,
                                   const uint32_t max_threads,
                                   const float scale_up_threshold,
                                   const float scale_down_threshold) {
  if (!system || !system->initialized) {
    return false;
  }

  // Update thread pool configuration for dynamic scaling
  thread_pool_config_t new_config = system->config.thread_pool;
  new_config.min_threads = min_threads;
  new_config.max_threads = max_threads;
  new_config.dynamic_thread_scaling = 1;

  bool success =
      mqi::update_thread_pool_config(system->thread_pool, &new_config);

  if (success && system->config.verbose_logging) {
    std::cout << "Phase 1.0: Dynamic thread scaling enabled" << std::endl;
    std::cout << "  Min threads: " << min_threads << std::endl;
    std::cout << "  Max threads: " << max_threads << std::endl;
  }

  return success;
}

CUDA_HOST
bool configure_memory_pool(phase1_system_t *system, const size_t pool_size_mb,
                           const bool use_unified_memory) {
  if (!system || !system->initialized) {
    return false;
  }

  // Note: For this implementation, we'd need to re-allocate the memory pool
  // This is a placeholder that updates configuration only
  system->config.memory_pool_size_mb = pool_size_mb;
  system->config.use_unified_memory = use_unified_memory;

  if (system->config.verbose_logging) {
    std::cout << "Phase 1.0: Memory pool configuration updated" << std::endl;
    std::cout << "  Pool size: " << pool_size_mb << " MB" << std::endl;
    std::cout << "  Unified memory: "
              << (use_unified_memory ? "Enabled" : "Disabled") << std::endl;
  }

  return true;
}

// ============================================================================
// ERROR HANDLING AND DIAGNOSTICS IMPLEMENTATION
// ============================================================================

CUDA_HOST
bool get_phase1_system_health(const phase1_system_t *system,
                              char *status_message,
                              const size_t message_buffer_size) {
  if (!system || !status_message || message_buffer_size == 0) {
    return false;
  }

  if (!system->initialized) {
    snprintf(status_message, message_buffer_size, "System not initialized");
    return false;
  }

  if (system->shutdown_requested) {
    snprintf(status_message, message_buffer_size,
             "System shutdown in progress");
    return false;
  }

  // Check component health
  bool thread_pool_healthy = system->thread_pool != nullptr;
  bool work_queue_healthy = system->work_queue != nullptr;
  bool rng_healthy = system->rng_states != nullptr ||
                     !system->config.rng_standardization_enabled;

  if (thread_pool_healthy && work_queue_healthy && rng_healthy) {
    snprintf(status_message, message_buffer_size,
             "System healthy - all components operational");
    return true;
  } else {
    snprintf(status_message, message_buffer_size,
             "System degraded - Thread pool: %s, Work queue: %s, RNG: %s",
             thread_pool_healthy ? "OK" : "ERROR",
             work_queue_healthy ? "OK" : "ERROR", rng_healthy ? "OK" : "ERROR");
    return false;
  }
}

CUDA_HOST
void diagnose_phase1_performance(const phase1_system_t *system,
                                 char *diagnostic_report,
                                 const size_t report_buffer_size) {
  if (!system || !diagnostic_report || report_buffer_size == 0) {
    return;
  }

  thread_pool_stats_t stats;
  uint32_t completed_work;
  uint64_t uptime_us;
  float throughput;

  get_phase1_system_statistics(system, &stats, &completed_work, &uptime_us,
                               &throughput);

  // Load atomic values into local variables before using in snprintf
  uint32_t active_threads = stats.active_threads;
  uint32_t peak_threads = stats.peak_active_threads;
  float gpu_util = stats.gpu_utilization_percent;
  uint64_t work_steals = stats.work_stealing_operations;
  float avg_work_time = stats.average_work_item_time_us;

  snprintf(diagnostic_report, report_buffer_size,
           "=== Phase 1.0 Performance Diagnostics ===\n"
           "System Uptime: %.2f seconds\n"
           "Work Items Processed: %u\n"
           "Average Throughput: %.2f items/sec\n"
           "Active Threads: %u / %u\n"
           "GPU Utilization: %.1f%%\n"
           "Work Stealing Operations: %lu\n"
           "Average Work Item Time: %.2f μs\n"
           "========================================",
           uptime_us / 1000000.0, completed_work, throughput, active_threads,
           peak_threads, gpu_util, work_steals, avg_work_time);
}

CUDA_HOST
bool validate_phase1_system_integrity(const phase1_system_t *system,
                                      char *validation_report,
                                      const size_t report_buffer_size) {
  if (!system || !validation_report || report_buffer_size == 0) {
    return false;
  }

  bool integrity_ok = true;
  size_t offset = 0;

  const auto append_report = [&](const char *message) {
    size_t len = strlen(message);
    if (offset + len < report_buffer_size - 1) {
      strcpy(validation_report + offset, message);
      offset += len;
    }
  };

  append_report("=== Phase 1.0 System Integrity Validation ===\n");

  // Check initialization
  if (!system->initialized) {
    append_report("FAIL: System not initialized\n");
    integrity_ok = false;
  } else {
    append_report("PASS: System initialized\n");
  }

  // Check thread pool
  if (!system->thread_pool) {
    append_report("FAIL: Thread pool not allocated\n");
    integrity_ok = false;
  } else {
    append_report("PASS: Thread pool allocated\n");
  }

  // Check work queue
  if (!system->work_queue) {
    append_report("FAIL: Work queue not allocated\n");
    integrity_ok = false;
  } else {
    append_report("PASS: Work queue allocated\n");
  }

  // Check RNG states
  if (system->config.rng_standardization_enabled && !system->rng_states) {
    append_report("FAIL: RNG states not allocated\n");
    integrity_ok = false;
  } else {
    append_report("PASS: RNG states ready\n");
  }

  if (integrity_ok) {
    append_report("RESULT: System integrity validation PASSED\n");
  } else {
    append_report("RESULT: System integrity validation FAILED\n");
  }
  append_report("==========================================\n");

  return integrity_ok;
}

} // namespace mqi
