#ifndef MQI_PHASE4_DYNAMIC_LOD_HPP
#define MQI_PHASE4_DYNAMIC_LOD_HPP

/// \file
///
/// Phase 4: Dynamic Level of Detail (LOD) System
///
/// This header implements dynamic LOD for geometry optimization, allowing
/// runtime adjustment of geometry complexity based on performance requirements.

#include "mqi_common.hpp"
#include "mqi_config.hpp"
#include "mqi_geometry_interface.hpp"
#include <array>
#include <memory>
#include <vector>
#include <mutex>

namespace mqi {

/// LOD configuration parameters
template <typename R> struct lod_config {
  R distance_threshold_coarse;   ///< Distance for switching to coarse LOD
  R distance_threshold_medium;   ///< Distance for switching to medium LOD
  R performance_target_ms;       ///< Target performance per intersection (ms)
  float quality_factor;          ///< Quality vs performance trade-off (0-1)
  bool enable_adaptive_lod;      ///< Enable adaptive LOD adjustment
  uint32_t lod_update_frequency; ///< Frames between LOD updates
  float memory_budget_mb;        ///< Memory budget for LOD (MB)

  CUDA_HOST_DEVICE
  lod_config()
      : distance_threshold_coarse(mqi::lod::DEFAULT_DISTANCE_THRESHOLD_COARSE),
        distance_threshold_medium(mqi::lod::DEFAULT_DISTANCE_THRESHOLD_MEDIUM),
        performance_target_ms(0.1f), quality_factor(0.8f),
        enable_adaptive_lod(true), lod_update_frequency(60),
        memory_budget_mb(100.0f) {}
};

/// LOD level definition
template <typename R> struct lod_level {
  geometry_complexity_t complexity; ///< Complexity level
  R max_view_distance;              ///< Maximum effective view distance
  float performance_cost;           ///< Relative performance cost
  float accuracy_estimate;          ///< Estimated accuracy
  size_t memory_usage;              ///< Memory usage in bytes
  std::unique_ptr<geometry_interface<R>>
      geometry_instance; ///< Geometry instance

  CUDA_HOST_DEVICE
  lod_level()
      : complexity(MEDIUM), max_view_distance(1000.0), performance_cost(1.0f),
        accuracy_estimate(0.9f), memory_usage(0) {}
};

/// LOD statistics for monitoring
template <typename R> struct lod_statistics {
  uint32_t total_frames;                   ///< Total frames processed
  uint32_t lod_changes;                    ///< Number of LOD changes
  float avg_performance_ms;                ///< Average performance
  float current_performance_ms;            ///< Current frame performance
  geometry_complexity_t current_lod;       ///< Current LOD level
  float memory_usage_mb;                   ///< Current memory usage
  std::array<uint32_t, 3> lod_usage_count; ///< Usage count per LOD level

  CUDA_HOST_DEVICE
  lod_statistics()
      : total_frames(0), lod_changes(0), avg_performance_ms(0.0f),
        current_performance_ms(0.0f), current_lod(MEDIUM),
        memory_usage_mb(0.0f) {
    lod_usage_count = {0, 0, 0};
  }
};

/// Dynamic LOD manager for geometry optimization
template <typename R> class dynamic_lod_manager {
private:
  std::vector<lod_level<R>> lod_levels_;
  lod_config<R> config_;
  lod_statistics<R> stats_;
  geometry_complexity_t current_lod_;
  uint32_t frame_counter_;
  std::vector<float> performance_history_;
  bool needs_update_;
  mutable std::mutex lod_mutex_;

public:
  /// Constructor
  dynamic_lod_manager();

  /// Configure LOD system
  void configure(const lod_config<R> &config);

  /// Add LOD level
  void add_lod_level(geometry_complexity_t complexity,
                     std::unique_ptr<geometry_interface<R>> geometry,
                     R max_view_distance, float performance_cost,
                     float accuracy_estimate);

  /// Get optimal LOD for given conditions
  geometry_complexity_t
  get_optimal_lod(const vec3<R> &view_position, const vec3<R> &object_position,
                  float current_performance_ms = 0.0f) const;

  /// Get geometry instance for current LOD
  geometry_interface<R> *get_current_geometry() const;

  /// Update LOD based on performance feedback
  void update_lod(const vec3<R> &view_position, const vec3<R> &object_position,
                  float frame_performance_ms);

  /// Force LOD level
  void set_lod(geometry_complexity_t lod);

  /// Get current LOD level
  geometry_complexity_t get_current_lod() const { return current_lod_; }

  /// Get LOD configuration
  const lod_config<R> &get_config() const { return config_; }

  /// Get LOD statistics
  const lod_statistics<R> &get_statistics() const { return stats_; }

  /// Reset statistics
  void reset_statistics();

  /// Check if LOD update is needed
  bool needs_lod_update() const { return needs_update_; }

  /// Update performance budget
  void update_performance_budget(float target_ms);

  /// Get memory usage estimate
  float get_memory_usage_mb() const;

  /// Validate LOD configuration
  bool validate_configuration() const;

private:
  /// Calculate optimal LOD based on distance
  geometry_complexity_t calculate_distance_based_lod(R distance) const;

  /// Calculate optimal LOD based on performance
  geometry_complexity_t
  calculate_performance_based_lod(float current_performance) const;

  /// Update statistics
  void update_statistics(float frame_performance_ms);

  /// Check if LOD transition is beneficial
  bool is_lod_transition_beneficial(geometry_complexity_t new_lod) const;
};

/// Adaptive LOD controller that automatically adjusts LOD
template <typename R> class adaptive_lod_controller {
private:
  dynamic_lod_manager<R> &lod_manager_;
  std::vector<float> performance_buffer_;
  size_t buffer_size_;
  float performance_threshold_;
  uint32_t adjustment_frequency_;
  uint32_t frame_counter_;

public:
  /// Constructor
  adaptive_lod_controller(dynamic_lod_manager<R> &manager,
                          size_t buffer_size = 120, // 2 seconds at 60 FPS
                          float performance_threshold = 0.1f,
                          uint32_t adjustment_frequency = 30);

  /// Update controller with new frame data
  void update(const vec3<R> &view_position, const vec3<R> &object_position,
              float frame_performance_ms);

  /// Set adaptive parameters
  void set_performance_threshold(float threshold) {
    performance_threshold_ = threshold;
  }
  void set_adjustment_frequency(uint32_t frequency) {
    adjustment_frequency_ = frequency;
  }
  void set_buffer_size(size_t size) { buffer_size_ = size; }

  /// Get average performance over buffer
  float get_average_performance() const;

  /// Get performance trend (positive = improving, negative = degrading)
  float get_performance_trend() const;

  /// Reset controller state
  void reset();

private:
  /// Calculate performance trend
  float calculate_trend() const;

  /// Determine if adjustment is needed
  bool needs_adjustment() const;
};

/// LOD-based geometry cache for efficient LOD switching
template <typename R> class lod_geometry_cache {
private:
  struct cache_entry {
    std::unique_ptr<geometry_interface<R>> geometry;
    geometry_complexity_t complexity;
    uint32_t access_count;
    uint32_t last_access_frame;
    float preparation_cost;
    bool is_prepared;
  };

  std::vector<cache_entry> cache_entries_;
  size_t max_cache_size_;
  size_t max_memory_mb_;
  uint32_t current_frame_;
  mutable std::mutex cache_mutex_;

public:
  /// Constructor
  lod_geometry_cache(size_t max_size = 10, size_t max_memory_mb = 500);

  /// Get or create geometry for LOD level
  geometry_interface<R> *
  get_or_create_geometry(geometry_complexity_t complexity,
                         std::function<std::unique_ptr<geometry_interface<R>>(
                             geometry_complexity_t)>
                             creator);

  /// Prepare geometry for GPU
  bool prepare_geometry(geometry_complexity_t complexity);

  /// Evict least recently used entries if needed
  void evict_lru_entries();

  /// Clear cache
  void clear_cache();

  /// Get cache statistics
  struct cache_stats {
    size_t total_entries;
    size_t prepared_entries;
    float memory_usage_mb;
    float hit_rate;
    uint32_t total_requests;
    uint32_t cache_hits;
  };

  cache_stats get_statistics() const;

  /// Set cache limits
  void set_max_size(size_t max_size) { max_cache_size_ = max_size; }
  void set_max_memory(size_t max_memory_mb) { max_memory_mb_ = max_memory_mb; }

private:
  /// Find cache entry by complexity
  cache_entry *find_entry(geometry_complexity_t complexity);

  /// Create new cache entry
  cache_entry *create_entry(geometry_complexity_t complexity,
                            std::unique_ptr<geometry_interface<R>> geometry);

  /// Update access information
  void update_access(cache_entry &entry);

  /// Calculate memory usage
  size_t calculate_memory_usage() const;
};

/// LOD optimization utilities
namespace lod_utils {
/// Calculate optimal LOD distances based on performance targets
template <typename R>
void calculate_optimal_lod_distances(
    const std::array<float, 3> &performance_costs,
    const std::array<float, 3> &accuracy_estimates, R target_performance_ms,
    std::array<R, 2> &distance_thresholds);

/// Estimate performance for given LOD level
template <typename R>
float estimate_lod_performance(geometry_complexity_t complexity, R distance,
                               const lod_config<R> &config);

/// Generate LOD transition matrix
template <typename R>
std::vector<std::vector<float>>
generate_lod_transition_matrix(const std::vector<lod_level<R>> &lod_levels);

/// Optimize LOD configuration using genetic algorithm
template <typename R>
lod_config<R> optimize_lod_configuration(
    const std::vector<lod_level<R>> &lod_levels,
    const std::vector<std::pair<vec3<R>, vec3<R>>> &test_scenarios,
    float target_performance_ms, size_t generations = 100);

} // namespace lod_utils

} // namespace mqi

#endif // MQI_PHASE4_DYNAMIC_LOD_HPP
