#ifndef MQI_PHASE4_CONFIG_HPP
#define MQI_PHASE4_CONFIG_HPP

/// \file
///
/// Phase 4: Performance/Accuracy Trade-off Configuration System
///
/// This header implements the configuration system for managing performance
/// and accuracy trade-offs in the Phase 4 geometry integration.

#include "mqi_common.hpp"
#include "mqi_dynamic_lod.hpp"
#include "mqi_geometry_interface.hpp"
#include <functional>
#include <string>
#include <unordered_map>
#include <mutex>

namespace mqi {

/// Performance profile presets
enum class performance_profile {
  REALTIME = 0,     ///< Maximum performance, lower accuracy
  INTERACTIVE = 1,  ///< Good performance, reasonable accuracy
  BALANCED = 2,     ///< Balanced performance and accuracy
  HIGH_QUALITY = 3, ///< High accuracy, moderate performance
  ULTRA_QUALITY = 4 ///< Maximum accuracy, lower performance
};

/// Quality level presets
enum class quality_level {
  LOW = 0,    ///< Fast approximate calculations
  MEDIUM = 1, ///< Standard quality
  HIGH = 2,   ///< High quality calculations
  ULTRA = 3   ///< Maximum quality
};

/// Application context for configuration tuning
enum class application_context {
  TREATMENT_PLANNING = 0, ///< Clinical treatment planning
  RESEARCH = 1,           ///< Research and development
  EDUCATION = 2,          ///< Educational purposes
  PROTOTYPING = 3,        ///< Rapid prototyping
  PRODUCTION = 4          ///< Clinical production
};

/// Phase 4 configuration parameters
template <typename R> struct phase4_config {
  // Core performance settings
  performance_profile profile;
  quality_level quality;
  application_context context;

  // Geometry settings
  geometry_type primary_geometry_type;
  geometry_precision_t default_precision;
  geometry_complexity_t default_complexity;
  bool enable_adaptive_precision;
  bool enable_adaptive_complexity;

  // LOD settings
  lod_config<R> lod_settings;
  bool enable_lod_caching;
  size_t max_lod_cache_size;
  float lod_transition_smoothness;

  // GPU settings
  bool enable_gpu_acceleration;
  bool enable_mixed_precision;
  size_t gpu_memory_budget_mb;
  bool enable_async_transfer;

  // Performance targets
  float target_intersection_time_ms;
  float target_frame_time_ms;
  float memory_budget_mb;
  float cpu_utilization_target;

  // Quality settings
  float minimum_accuracy_threshold;
  float maximum_acceptable_error;
  bool enable_error_estimation;
  bool enable_quality_metrics;

  // Plugin settings
  std::string preferred_geometry_plugin;
  std::vector<std::string> enabled_plugins;
  bool enable_plugin_profiling;
  bool enable_auto_plugin_selection;

  // Debug and monitoring
  bool enable_performance_monitoring;
  bool enable_debug_visualization;
  bool enable_statistics_logging;
  uint32_t statistics_update_interval;

  // Fallback settings
  bool enable_fallback_mechanisms;
  std::string fallback_geometry_plugin;
  geometry_precision_t fallback_precision;
  float fallback_performance_threshold;

  // Advanced settings
  bool enable_predictive_loading;
  bool enable_background_optimization;
  uint32_t optimization_thread_count;
  float quality_adaptation_rate;

  // Constructor with defaults
  phase4_config()
      : profile(performance_profile::BALANCED), quality(quality_level::MEDIUM),
        context(application_context::TREATMENT_PLANNING),
        primary_geometry_type(geometry_type::PATIENT),
        default_precision(geometry_precision_t::BALANCED),
        default_complexity(geometry_complexity_t::MEDIUM),
        enable_adaptive_precision(true), enable_adaptive_complexity(true),
        enable_lod_caching(true), max_lod_cache_size(10),
        lod_transition_smoothness(0.8f), enable_gpu_acceleration(true),
        enable_mixed_precision(false), gpu_memory_budget_mb(1024),
        enable_async_transfer(true), target_intersection_time_ms(0.1f),
        target_frame_time_ms(16.67f), // 60 FPS
        memory_budget_mb(2048), cpu_utilization_target(0.8f),
        minimum_accuracy_threshold(0.95f), maximum_acceptable_error(0.05f),
        enable_error_estimation(true), enable_quality_metrics(true),
        preferred_geometry_plugin("voxel_bvh_hybrid"),
        enabled_plugins({"voxel_bvh_hybrid", "grid_based"}),
        enable_plugin_profiling(true), enable_auto_plugin_selection(true),
        enable_performance_monitoring(true), enable_debug_visualization(false),
        enable_statistics_logging(true), statistics_update_interval(60),
        enable_fallback_mechanisms(true),
        fallback_geometry_plugin("grid_based"),
        fallback_precision(geometry_precision_t::BALANCED),
        fallback_performance_threshold(2.0f), enable_predictive_loading(false),
        enable_background_optimization(true), optimization_thread_count(4),
        quality_adaptation_rate(0.1f) {}
};

/// Configuration manager for Phase 4
template <typename R> class phase4_config_manager {
private:
  phase4_config<R> current_config_;
  std::unordered_map<std::string, phase4_config<R>> preset_configs_;
  std::vector<std::function<void(const phase4_config<R> &)>>
      config_change_callbacks_;
  mutable std::mutex config_mutex_;

public:
  /// Constructor
  phase4_config_manager();

  /// Load configuration from preset
  bool load_preset(const std::string &preset_name);

  /// Save current configuration as preset
  bool save_preset(const std::string &preset_name,
                   const phase4_config<R> &config);

  /// Get current configuration
  const phase4_config<R> &get_config() const;

  /// Update configuration
  void update_config(const phase4_config<R> &new_config);

  /// Update specific configuration parameters
  void update_performance_profile(performance_profile profile);
  void update_quality_level(quality_level quality);
  void update_application_context(application_context context);

  /// Validate configuration
  bool validate_config(const phase4_config<R> &config) const;

  /// Get optimal configuration for given requirements
  phase4_config<R> get_optimal_config(float target_performance_ms,
                                      float target_accuracy,
                                      application_context context) const;

  /// Auto-tune configuration based on performance feedback
  void auto_tune_config(float current_performance_ms, float current_accuracy);

  /// Register configuration change callback
  void
  register_callback(std::function<void(const phase4_config<R> &)> callback);

  /// Export configuration to file
  bool export_config(const std::string &filename) const;

  /// Import configuration from file
  bool import_config(const std::string &filename);

  /// Get list of available presets
  std::vector<std::string> get_available_presets() const;

private:
  /// Initialize default presets
  void initialize_presets();

  /// Apply configuration changes
  void apply_config_changes(const phase4_config<R> &new_config);

  /// Notify callbacks of configuration change
  void notify_callbacks(const phase4_config<R> &config);
};

/// Configuration validator and optimizer
template <typename R> class config_optimizer {
public:
  /// Optimization constraints
  struct optimization_constraints {
    float min_performance_ms;
    float max_performance_ms;
    float min_accuracy;
    float max_memory_mb;
    bool require_gpu_support;
    std::vector<std::string> required_plugins;
  };

  /// Optimization result
  struct optimization_result {
    phase4_config<R> optimal_config;
    float predicted_performance_ms;
    float predicted_accuracy;
    float predicted_memory_mb;
    bool meets_constraints;
    std::vector<std::string> recommendations;
  };

  /// Optimize configuration for given constraints
  optimization_result
  optimize_config(const optimization_constraints &constraints,
                  const phase4_config<R> &initial_config) const;

  /// Predict performance for given configuration
  float predict_performance(const phase4_config<R> &config) const;

  /// Predict accuracy for given configuration
  float predict_accuracy(const phase4_config<R> &config) const;

  /// Predict memory usage for given configuration
  float predict_memory_usage(const phase4_config<R> &config) const;

  /// Generate configuration recommendations
  std::vector<std::string>
  generate_recommendations(const phase4_config<R> &config,
                           float current_performance_ms,
                           float current_accuracy) const;

private:
  /// Build performance model
  float calculate_performance_score(const phase4_config<R> &config) const;

  /// Build accuracy model
  float calculate_accuracy_score(const phase4_config<R> &config) const;

  /// Build memory model
  float calculate_memory_score(const phase4_config<R> &config) const;
};

/// Configuration presets factory
namespace config_presets {
/// Get configuration for specific performance profile
template <typename R>
phase4_config<R> get_profile_config(performance_profile profile);

/// Get configuration for specific application context
template <typename R>
phase4_config<R> get_context_config(application_context context);

/// Get configuration for specific quality level
template <typename R>
phase4_config<R> get_quality_config(quality_level quality);

/// Create custom configuration
template <typename R>
phase4_config<R> create_custom_config(
    performance_profile profile = performance_profile::BALANCED,
    quality_level quality = quality_level::MEDIUM,
    application_context context = application_context::TREATMENT_PLANNING);

} // namespace config_presets

} // namespace mqi

#endif // MQI_PHASE4_CONFIG_HPP
