#include "mqi_config.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

namespace mqi {

// phase4_config_manager implementation
template <typename R> phase4_config_manager<R>::phase4_config_manager() {
  initialize_presets();
}

template <typename R>
bool phase4_config_manager<R>::load_preset(const std::string &preset_name) {
  std::lock_guard<std::mutex> lock(config_mutex_);

  auto it = preset_configs_.find(preset_name);
  if (it != preset_configs_.end()) {
    current_config_ = it->second;
    apply_config_changes(current_config_);
    return true;
  }

  return false;
}

template <typename R>
bool phase4_config_manager<R>::save_preset(const std::string &preset_name,
                                           const phase4_config<R> &config) {
  if (!validate_config(config)) {
    return false;
  }

  std::lock_guard<std::mutex> lock(config_mutex_);
  preset_configs_[preset_name] = config;
  return true;
}

template <typename R>
const phase4_config<R> &phase4_config_manager<R>::get_config() const {
  std::lock_guard<std::mutex> lock(config_mutex_);
  return current_config_;
}

template <typename R>
void phase4_config_manager<R>::update_config(
    const phase4_config<R> &new_config) {
  if (!validate_config(new_config)) {
    return;
  }

  std::lock_guard<std::mutex> lock(config_mutex_);
  current_config_ = new_config;
  apply_config_changes(new_config);
  notify_callbacks(new_config);
}

template <typename R>
void phase4_config_manager<R>::update_performance_profile(
    performance_profile profile) {
  std::lock_guard<std::mutex> lock(config_mutex_);
  current_config_.profile = profile;
  apply_config_changes(current_config_);
  notify_callbacks(current_config_);
}

template <typename R>
void phase4_config_manager<R>::update_quality_level(quality_level quality) {
  std::lock_guard<std::mutex> lock(config_mutex_);
  current_config_.quality = quality;
  apply_config_changes(current_config_);
  notify_callbacks(current_config_);
}

template <typename R>
void phase4_config_manager<R>::update_application_context(
    application_context context) {
  std::lock_guard<std::mutex> lock(config_mutex_);
  current_config_.context = context;
  apply_config_changes(current_config_);
  notify_callbacks(current_config_);
}

template <typename R>
bool phase4_config_manager<R>::validate_config(
    const phase4_config<R> &config) const {
  // Validate performance targets
  if (config.target_intersection_time_ms <= 0 ||
      config.target_frame_time_ms <= 0) {
    return false;
  }

  // Validate memory budgets
  if (config.memory_budget_mb <= 0 || config.gpu_memory_budget_mb <= 0) {
    return false;
  }

  // Validate quality thresholds
  if (config.minimum_accuracy_threshold <= 0 ||
      config.minimum_accuracy_threshold > 1.0) {
    return false;
  }

  if (config.maximum_acceptable_error < 0 ||
      config.maximum_acceptable_error > 1.0) {
    return false;
  }

  // Validate LOD settings
  if (config.lod_settings.distance_threshold_coarse <= 0 ||
      config.lod_settings.distance_threshold_medium <= 0) {
    return false;
  }

  // Validate plugin configuration
  if (config.preferred_geometry_plugin.empty() &&
      config.enabled_plugins.empty()) {
    return false;
  }

  return true;
}

template <typename R>
phase4_config<R> phase4_config_manager<R>::get_optimal_config(
    float target_performance_ms, float target_accuracy,
    application_context context) const {
  phase4_config<R> config;
  config.context = context;

  // Determine performance profile based on targets
  if (target_performance_ms < 0.05f) {
    config.profile = performance_profile::REALTIME;
    config.quality = quality_level::LOW;
  } else if (target_performance_ms < 0.1f) {
    config.profile = performance_profile::INTERACTIVE;
    config.quality = quality_level::MEDIUM;
  } else if (target_performance_ms < 0.5f) {
    config.profile = performance_profile::BALANCED;
    config.quality = quality_level::HIGH;
  } else {
    config.profile = performance_profile::HIGH_QUALITY;
    config.quality = quality_level::ULTRA;
  }

  // Adjust based on accuracy requirements
  if (target_accuracy > 0.99f) {
    config.quality = quality_level::ULTRA;
    config.profile = performance_profile::HIGH_QUALITY;
  } else if (target_accuracy > 0.95f) {
    config.quality = quality_level::HIGH;
  } else if (target_accuracy < 0.8f) {
    config.quality = quality_level::LOW;
  }

  // Configure based on application context
  switch (context) {
  case application_context::TREATMENT_PLANNING:
    config.minimum_accuracy_threshold = 0.98f;
    config.enable_error_estimation = true;
    config.enable_fallback_mechanisms = true;
    break;
  case application_context::RESEARCH:
    config.minimum_accuracy_threshold = 0.95f;
    config.enable_debug_visualization = true;
    config.enable_statistics_logging = true;
    break;
  case application_context::EDUCATION:
    config.minimum_accuracy_threshold = 0.9f;
    config.enable_debug_visualization = true;
    config.profile = performance_profile::INTERACTIVE;
    break;
  case application_context::PROTOTYPING:
    config.minimum_accuracy_threshold = 0.8f;
    config.enable_performance_monitoring = true;
    config.profile = performance_profile::BALANCED;
    break;
  case application_context::PRODUCTION:
    config.minimum_accuracy_threshold = 0.99f;
    config.enable_fallback_mechanisms = true;
    config.enable_performance_monitoring = true;
    break;
  }

  // Set performance targets
  config.target_intersection_time_ms = target_performance_ms;
  config.minimum_accuracy_threshold = target_accuracy;

  // Configure geometry settings
  switch (config.profile) {
  case performance_profile::REALTIME:
    config.default_precision = geometry_precision_t::FAST_APPROXIMATION;
    config.default_complexity = geometry_complexity_t::COARSE;
    break;
  case performance_profile::INTERACTIVE:
    config.default_precision = geometry_precision_t::BALANCED;
    config.default_complexity = geometry_complexity_t::MEDIUM;
    break;
  case performance_profile::BALANCED:
    config.default_precision = geometry_precision_t::BALANCED;
    config.default_complexity = geometry_complexity_t::MEDIUM;
    break;
  case performance_profile::HIGH_QUALITY:
    config.default_precision = geometry_precision_t::HIGH_PRECISION;
    config.default_complexity = geometry_complexity_t::FINE;
    break;
  case performance_profile::ULTRA_QUALITY:
    config.default_precision = geometry_precision_t::HIGH_PRECISION;
    config.default_complexity = geometry_complexity_t::FINE;
    break;
  }

  return config;
}

template <typename R>
void phase4_config_manager<R>::auto_tune_config(float current_performance_ms,
                                                float current_accuracy) {
  std::lock_guard<std::mutex> lock(config_mutex_);

  bool config_changed = false;

  // Tune based on performance
  if (current_performance_ms >
      current_config_.target_intersection_time_ms * 1.5f) {
    // Performance is poor, reduce quality
    if (current_config_.profile < performance_profile::ULTRA_QUALITY) {
      current_config_.profile = static_cast<performance_profile>(
          static_cast<int>(current_config_.profile) + 1);
      config_changed = true;
    }
  } else if (current_performance_ms <
             current_config_.target_intersection_time_ms * 0.5f) {
    // Performance is good, can increase quality
    if (current_config_.profile > performance_profile::REALTIME) {
      current_config_.profile = static_cast<performance_profile>(
          static_cast<int>(current_config_.profile) - 1);
      config_changed = true;
    }
  }

  // Tune based on accuracy
  if (current_accuracy < current_config_.minimum_accuracy_threshold) {
    // Accuracy is insufficient, increase quality
    if (current_config_.quality < quality_level::ULTRA) {
      current_config_.quality = static_cast<quality_level>(
          static_cast<int>(current_config_.quality) + 1);
      config_changed = true;
    }
  } else if (current_accuracy >
             current_config_.minimum_accuracy_threshold * 1.1f) {
    // Accuracy is better than needed, can optimize for performance
    if (current_config_.quality > quality_level::LOW) {
      current_config_.quality = static_cast<quality_level>(
          static_cast<int>(current_config_.quality) - 1);
      config_changed = true;
    }
  }

  if (config_changed) {
    apply_config_changes(current_config_);
    notify_callbacks(current_config_);
  }
}

template <typename R>
void phase4_config_manager<R>::register_callback(
    std::function<void(const phase4_config<R> &)> callback) {
  config_change_callbacks_.push_back(callback);
}

template <typename R>
bool phase4_config_manager<R>::export_config(
    const std::string &filename) const {
  std::lock_guard<std::mutex> lock(config_mutex_);

  std::ofstream file(filename);
  if (!file.is_open()) {
    return false;
  }

  // Simple export format (could be JSON, XML, etc.)
  file << "# Moqui Phase 4 Configuration Export\n";
  file << "profile=" << static_cast<int>(current_config_.profile) << "\n";
  file << "quality=" << static_cast<int>(current_config_.quality) << "\n";
  file << "context=" << static_cast<int>(current_config_.context) << "\n";
  file << "target_performance_ms="
       << current_config_.target_intersection_time_ms << "\n";
  file << "minimum_accuracy=" << current_config_.minimum_accuracy_threshold
       << "\n";
  file << "enable_gpu="
       << (current_config_.enable_gpu_acceleration ? "true" : "false") << "\n";
  file << "preferred_plugin=" << current_config_.preferred_geometry_plugin
       << "\n";

  return true;
}

template <typename R>
bool phase4_config_manager<R>::import_config(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    return false;
  }

  phase4_config<R> config = current_config_;
  std::string line;

  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#')
      continue;

    size_t pos = line.find('=');
    if (pos != std::string::npos) {
      std::string key = line.substr(0, pos);
      std::string value = line.substr(pos + 1);

      if (key == "profile") {
        config.profile = static_cast<performance_profile>(std::stoi(value));
      } else if (key == "quality") {
        config.quality = static_cast<quality_level>(std::stoi(value));
      } else if (key == "context") {
        config.context = static_cast<application_context>(std::stoi(value));
      } else if (key == "target_performance_ms") {
        config.target_intersection_time_ms = std::stof(value);
      } else if (key == "minimum_accuracy") {
        config.minimum_accuracy_threshold = std::stof(value);
      } else if (key == "enable_gpu") {
        config.enable_gpu_acceleration = (value == "true");
      } else if (key == "preferred_plugin") {
        config.preferred_geometry_plugin = value;
      }
    }
  }

  if (validate_config(config)) {
    update_config(config);
    return true;
  }

  return false;
}

template <typename R>
std::vector<std::string>
phase4_config_manager<R>::get_available_presets() const {
  std::lock_guard<std::mutex> lock(config_mutex_);

  std::vector<std::string> presets;
  presets.reserve(preset_configs_.size());

  for (const auto &pair : preset_configs_) {
    presets.push_back(pair.first);
  }

  return presets;
}

template <typename R> void phase4_config_manager<R>::initialize_presets() {
  // Treatment planning preset
  {
    phase4_config<R> config;
    config.context = application_context::TREATMENT_PLANNING;
    config.profile = performance_profile::BALANCED;
    config.quality = quality_level::HIGH;
    config.minimum_accuracy_threshold = 0.98f;
    config.enable_error_estimation = true;
    config.enable_fallback_mechanisms = true;
    preset_configs_["treatment_planning"] = config;
  }

  // Real-time visualization preset
  {
    phase4_config<R> config;
    config.context = application_context::EDUCATION;
    config.profile = performance_profile::INTERACTIVE;
    config.quality = quality_level::MEDIUM;
    config.minimum_accuracy_threshold = 0.9f;
    config.enable_debug_visualization = true;
    preset_configs_["realtime_viz"] = config;
  }

  // Research preset
  {
    phase4_config<R> config;
    config.context = application_context::RESEARCH;
    config.profile = performance_profile::HIGH_QUALITY;
    config.quality = quality_level::ULTRA;
    config.minimum_accuracy_threshold = 0.95f;
    config.enable_debug_visualization = true;
    config.enable_statistics_logging = true;
    preset_configs_["research"] = config;
  }

  // Fast preview preset
  {
    phase4_config<R> config;
    config.context = application_context::PROTOTYPING;
    config.profile = performance_profile::REALTIME;
    config.quality = quality_level::LOW;
    config.minimum_accuracy_threshold = 0.8f;
    config.default_precision = geometry_precision_t::FAST_APPROXIMATION;
    config.default_complexity = geometry_complexity_t::COARSE;
    preset_configs_["fast_preview"] = config;
  }
}

template <typename R>
void phase4_config_manager<R>::apply_config_changes(
    const phase4_config<R> &new_config) {
  // This would apply the configuration to the actual system
  // For now, it's a placeholder for where the configuration would be applied
}

template <typename R>
void phase4_config_manager<R>::notify_callbacks(
    const phase4_config<R> &config) {
  for (auto &callback : config_change_callbacks_) {
    callback(config);
  }
}

// config_optimizer implementation
template <typename R>
typename config_optimizer<R>::optimization_result
config_optimizer<R>::optimize_config(
    const optimization_constraints &constraints,
    const phase4_config<R> &initial_config) const {
  optimization_result result;
  result.optimal_config = initial_config;
  result.meets_constraints = false;

  // Simple optimization: try different profiles and find the best that meets
  // constraints
  std::vector<performance_profile> profiles = {
      performance_profile::REALTIME, performance_profile::INTERACTIVE,
      performance_profile::BALANCED, performance_profile::HIGH_QUALITY,
      performance_profile::ULTRA_QUALITY};

  float best_score = -1.0f;

  for (auto profile : profiles) {
    phase4_config<R> test_config = initial_config;
    test_config.profile = profile;

    // Update geometry settings based on profile
    switch (profile) {
    case performance_profile::REALTIME:
      test_config.default_precision = geometry_precision_t::FAST_APPROXIMATION;
      test_config.default_complexity = geometry_complexity_t::COARSE;
      break;
    case performance_profile::INTERACTIVE:
      test_config.default_precision = geometry_precision_t::BALANCED;
      test_config.default_complexity = geometry_complexity_t::MEDIUM;
      break;
    case performance_profile::BALANCED:
      test_config.default_precision = geometry_precision_t::BALANCED;
      test_config.default_complexity = geometry_complexity_t::MEDIUM;
      break;
    case performance_profile::HIGH_QUALITY:
      test_config.default_precision = geometry_precision_t::HIGH_PRECISION;
      test_config.default_complexity = geometry_complexity_t::FINE;
      break;
    case performance_profile::ULTRA_QUALITY:
      test_config.default_precision = geometry_precision_t::HIGH_PRECISION;
      test_config.default_complexity = geometry_complexity_t::FINE;
      break;
    }

    float predicted_perf = predict_performance(test_config);
    float predicted_acc = predict_accuracy(test_config);
    float predicted_mem = predict_memory_usage(test_config);

    // Check if constraints are met
    bool meets_constraints =
        (predicted_perf >= constraints.min_performance_ms &&
         predicted_perf <= constraints.max_performance_ms &&
         predicted_acc >= constraints.min_accuracy &&
         predicted_mem <= constraints.max_memory_mb);

    // Calculate overall score
    float perf_score =
        1.0f /
        (1.0f + std::abs(predicted_perf - constraints.min_performance_ms));
    float acc_score = predicted_acc;
    float mem_score = 1.0f - (predicted_mem / constraints.max_memory_mb);
    float total_score = (perf_score + acc_score + mem_score) / 3.0f;

    if (meets_constraints && total_score > best_score) {
      best_score = total_score;
      result.optimal_config = test_config;
      result.predicted_performance_ms = predicted_perf;
      result.predicted_accuracy = predicted_acc;
      result.predicted_memory_mb = predicted_mem;
      result.meets_constraints = true;
    }
  }

  if (!result.meets_constraints) {
    result.recommendations.push_back("No configuration meets all constraints");
    result.recommendations.push_back(
        "Consider relaxing performance or memory requirements");
  }

  return result;
}

template <typename R>
float config_optimizer<R>::predict_performance(
    const phase4_config<R> &config) const {
  return calculate_performance_score(config);
}

template <typename R>
float config_optimizer<R>::predict_accuracy(
    const phase4_config<R> &config) const {
  return calculate_accuracy_score(config);
}

template <typename R>
float config_optimizer<R>::predict_memory_usage(
    const phase4_config<R> &config) const {
  return calculate_memory_score(config);
}

template <typename R>
std::vector<std::string>
config_optimizer<R>::generate_recommendations(const phase4_config<R> &config,
                                              float current_performance_ms,
                                              float current_accuracy) const {
  std::vector<std::string> recommendations;

  if (current_performance_ms > config.target_intersection_time_ms * 1.5f) {
    recommendations.push_back(
        "Consider using a lower quality profile for better performance");
    recommendations.push_back("Enable adaptive precision and complexity");
    if (!config.enable_gpu_acceleration) {
      recommendations.push_back("Enable GPU acceleration if available");
    }
  }

  if (current_accuracy < config.minimum_accuracy_threshold) {
    recommendations.push_back(
        "Increase quality level to meet accuracy requirements");
    recommendations.push_back("Use higher precision geometry settings");
    recommendations.push_back("Enable error estimation and correction");
  }

  if (config.memory_budget_mb > 1024 &&
      current_performance_ms > config.target_intersection_time_ms) {
    recommendations.push_back(
        "Consider reducing memory budget to improve cache performance");
  }

  return recommendations;
}

template <typename R>
float config_optimizer<R>::calculate_performance_score(
    const phase4_config<R> &config) const {
  float base_score = 1.0f;

  // Adjust based on profile
  switch (config.profile) {
  case performance_profile::REALTIME:
    base_score *= 0.2f;
    break;
  case performance_profile::INTERACTIVE:
    base_score *= 0.5f;
    break;
  case performance_profile::BALANCED:
    base_score *= 1.0f;
    break;
  case performance_profile::HIGH_QUALITY:
    base_score *= 2.0f;
    break;
  case performance_profile::ULTRA_QUALITY:
    base_score *= 4.0f;
    break;
  }

  // Adjust based on precision
  switch (config.default_precision) {
  case geometry_precision_t::FAST_APPROXIMATION:
    base_score *= 0.5f;
    break;
  case geometry_precision_t::BALANCED:
    base_score *= 1.0f;
    break;
  case geometry_precision_t::HIGH_PRECISION:
    base_score *= 2.0f;
    break;
  }

  // Adjust based on complexity
  switch (config.default_complexity) {
  case geometry_complexity_t::COARSE:
    base_score *= 0.3f;
    break;
  case geometry_complexity_t::MEDIUM:
    base_score *= 1.0f;
    break;
  case geometry_complexity_t::FINE:
    base_score *= 3.0f;
    break;
  }

  // GPU acceleration
  if (config.enable_gpu_acceleration) {
    base_score *= 0.7f; // Better performance with GPU
  }

  return base_score * config.target_intersection_time_ms;
}

template <typename R>
float config_optimizer<R>::calculate_accuracy_score(
    const phase4_config<R> &config) const {
  float base_score = 0.9f;

  // Adjust based on quality level
  switch (config.quality) {
  case quality_level::LOW:
    base_score *= 0.7f;
    break;
  case quality_level::MEDIUM:
    base_score *= 0.85f;
    break;
  case quality_level::HIGH:
    base_score *= 0.95f;
    break;
  case quality_level::ULTRA:
    base_score *= 0.99f;
    break;
  }

  // Adjust based on precision
  switch (config.default_precision) {
  case geometry_precision_t::FAST_APPROXIMATION:
    base_score *= 0.8f;
    break;
  case geometry_precision_t::BALANCED:
    base_score *= 0.95f;
    break;
  case geometry_precision_t::HIGH_PRECISION:
    base_score *= 0.99f;
    break;
  }

  // Adjust based on complexity
  switch (config.default_complexity) {
  case geometry_complexity_t::COARSE:
    base_score *= 0.8f;
    break;
  case geometry_complexity_t::MEDIUM:
    base_score *= 0.95f;
    break;
  case geometry_complexity_t::FINE:
    base_score *= 0.99f;
    break;
  }

  // Error estimation improves accuracy
  if (config.enable_error_estimation) {
    base_score *= 1.02f;
  }

  return std::min(base_score, 1.0f);
}

template <typename R>
float config_optimizer<R>::calculate_memory_score(
    const phase4_config<R> &config) const {
  float base_memory = 100.0f; // Base memory usage in MB

  // Adjust based on complexity
  switch (config.default_complexity) {
  case geometry_complexity_t::COARSE:
    base_memory *= 0.5f;
    break;
  case geometry_complexity_t::MEDIUM:
    base_memory *= 1.0f;
    break;
  case geometry_complexity_t::FINE:
    base_memory *= 2.5f;
    break;
  }

  // LOD caching
  if (config.enable_lod_caching) {
    base_memory +=
        config.max_lod_cache_size * 10.0f; // Additional memory for cache
  }

  // GPU memory
  if (config.enable_gpu_acceleration) {
    base_memory += config.gpu_memory_budget_mb;
  }

  return base_memory;
}

// Presets factory implementation
namespace config_presets {
template <typename R>
phase4_config<R> get_profile_config(performance_profile profile) {
  phase4_config<R> config;
  config.profile = profile;

  switch (profile) {
  case performance_profile::REALTIME:
    config.quality = quality_level::LOW;
    config.target_intersection_time_ms = 0.05f;
    config.default_precision = geometry_precision_t::FAST_APPROXIMATION;
    config.default_complexity = geometry_complexity_t::COARSE;
    break;
  case performance_profile::INTERACTIVE:
    config.quality = quality_level::MEDIUM;
    config.target_intersection_time_ms = 0.1f;
    config.default_precision = geometry_precision_t::BALANCED;
    config.default_complexity = geometry_complexity_t::MEDIUM;
    break;
  case performance_profile::BALANCED:
    config.quality = quality_level::HIGH;
    config.target_intersection_time_ms = 0.2f;
    config.default_precision = geometry_precision_t::BALANCED;
    config.default_complexity = geometry_complexity_t::MEDIUM;
    break;
  case performance_profile::HIGH_QUALITY:
    config.quality = quality_level::HIGH;
    config.target_intersection_time_ms = 0.5f;
    config.default_precision = geometry_precision_t::HIGH_PRECISION;
    config.default_complexity = geometry_complexity_t::FINE;
    break;
  case performance_profile::ULTRA_QUALITY:
    config.quality = quality_level::ULTRA;
    config.target_intersection_time_ms = 1.0f;
    config.default_precision = geometry_precision_t::HIGH_PRECISION;
    config.default_complexity = geometry_complexity_t::FINE;
    break;
  }

  return config;
}

template <typename R>
phase4_config<R> create_custom_config(performance_profile profile,
                                      quality_level quality,
                                      application_context context) {
  phase4_config<R> config;
  config.profile = profile;
  config.quality = quality;
  config.context = context;

  // Set appropriate defaults based on the combination
  if (profile == performance_profile::REALTIME ||
      quality == quality_level::LOW) {
    config.default_precision = geometry_precision_t::FAST_APPROXIMATION;
    config.default_complexity = geometry_complexity_t::COARSE;
  } else if (profile == performance_profile::ULTRA_QUALITY ||
             quality == quality_level::ULTRA) {
    config.default_precision = geometry_precision_t::HIGH_PRECISION;
    config.default_complexity = geometry_complexity_t::FINE;
  } else {
    config.default_precision = geometry_precision_t::BALANCED;
    config.default_complexity = geometry_complexity_t::MEDIUM;
  }

  return config;
}
} // namespace config_presets

// Explicit template instantiations
template class phase4_config_manager<float>;
template class phase4_config_manager<double>;
template class config_optimizer<float>;
template class config_optimizer<double>;

} // namespace mqi
