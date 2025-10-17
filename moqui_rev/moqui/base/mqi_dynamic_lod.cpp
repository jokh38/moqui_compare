#include "mqi_dynamic_lod.hpp"
#include "mqi_math.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>
#include <numeric>

namespace mqi {

// dynamic_lod_manager implementation
template <typename R>
dynamic_lod_manager<R>::dynamic_lod_manager()
    : current_lod_(MEDIUM), frame_counter_(0), needs_update_(false) {
  performance_history_.reserve(1000);
  lod_levels_.reserve(3);
  stats_ = lod_statistics<R>();
}

template <typename R>
void dynamic_lod_manager<R>::configure(const lod_config<R> &config) {
  std::lock_guard<std::mutex> lock(lod_mutex_);
  config_ = config;
  needs_update_ = true;
}

template <typename R>
void dynamic_lod_manager<R>::add_lod_level(
    geometry_complexity_t complexity,
    std::unique_ptr<geometry_interface<R>> geometry, R max_view_distance,
    float performance_cost, float accuracy_estimate) {
  std::lock_guard<std::mutex> lock(lod_mutex_);

  lod_level<R> level;
  level.complexity = complexity;
  level.geometry_instance = std::move(geometry);
  level.max_view_distance = max_view_distance;
  level.performance_cost = performance_cost;
  level.accuracy_estimate = accuracy_estimate;
  level.memory_usage = level.geometry_instance
                           ? level.geometry_instance->estimate_memory_usage()
                           : 0;

  lod_levels_.push_back(std::move(level));

  // Sort by complexity (coarse to fine)
  std::sort(lod_levels_.begin(), lod_levels_.end(),
            [](const lod_level<R> &a, const lod_level<R> &b) {
              return static_cast<int>(a.complexity) <
                     static_cast<int>(b.complexity);
            });

  needs_update_ = true;
}

template <typename R>
geometry_complexity_t
dynamic_lod_manager<R>::get_optimal_lod(const vec3<R> &view_position,
                                        const vec3<R> &object_position,
                                        float current_performance_ms) const {
  std::lock_guard<std::mutex> lock(lod_mutex_);

  R distance = (view_position - object_position).norm();

  // Start with distance-based LOD
  geometry_complexity_t distance_lod = calculate_distance_based_lod(distance);

  // Adjust based on performance if adaptive LOD is enabled
  if (config_.enable_adaptive_lod && current_performance_ms > 0) {
    geometry_complexity_t performance_lod =
        calculate_performance_based_lod(current_performance_ms);

    // Choose the coarser LOD (lower complexity) to meet performance targets
    return static_cast<geometry_complexity_t>(std::min(
        static_cast<int>(distance_lod), static_cast<int>(performance_lod)));
  }

  return distance_lod;
}

template <typename R>
geometry_interface<R> *dynamic_lod_manager<R>::get_current_geometry() const {
  std::lock_guard<std::mutex> lock(lod_mutex_);

  auto it = std::find_if(lod_levels_.begin(), lod_levels_.end(),
                         [this](const lod_level<R> &level) {
                           return level.complexity == current_lod_;
                         });

  return (it != lod_levels_.end() && it->geometry_instance)
             ? it->geometry_instance.get()
             : nullptr;
}

template <typename R>
void dynamic_lod_manager<R>::update_lod(const vec3<R> &view_position,
                                        const vec3<R> &object_position,
                                        float frame_performance_ms) {
  frame_counter_++;

  // Update performance history
  if (frame_performance_ms > 0) {
    performance_history_.push_back(frame_performance_ms);
    if (performance_history_.size() > 1000) {
      performance_history_.erase(performance_history_.begin());
    }
  }

  // Check if we should update LOD
  bool should_update =
      (frame_counter_ % config_.lod_update_frequency == 0) || needs_update_ ||
      (config_.enable_adaptive_lod &&
       frame_performance_ms > config_.performance_target_ms * 1.5f);

  if (should_update) {
    geometry_complexity_t optimal_lod =
        get_optimal_lod(view_position, object_position, frame_performance_ms);

    if (optimal_lod != current_lod_ &&
        is_lod_transition_beneficial(optimal_lod)) {
      set_lod(optimal_lod);
      stats_.lod_changes++;
    }

    needs_update_ = false;
  }

  update_statistics(frame_performance_ms);
}

template <typename R>
void dynamic_lod_manager<R>::set_lod(geometry_complexity_t lod) {
  std::lock_guard<std::mutex> lock(lod_mutex_);

  if (lod != current_lod_) {
    current_lod_ = lod;
    stats_.current_lod = lod;

    // Update LOD usage count
    if (static_cast<int>(lod) >= 0 && static_cast<int>(lod) < 3) {
      stats_.lod_usage_count[static_cast<int>(lod)]++;
    }
  }
}

template <typename R> void dynamic_lod_manager<R>::reset_statistics() {
  std::lock_guard<std::mutex> lock(lod_mutex_);
  stats_ = lod_statistics<R>();
  performance_history_.clear();
  frame_counter_ = 0;
}

template <typename R>
void dynamic_lod_manager<R>::update_performance_budget(float target_ms) {
  std::lock_guard<std::mutex> lock(lod_mutex_);
  config_.performance_target_ms = target_ms;
  needs_update_ = true;
}

template <typename R>
float dynamic_lod_manager<R>::get_memory_usage_mb() const {
  std::lock_guard<std::mutex> lock(lod_mutex_);

  size_t total_memory = 0;
  for (const auto &level : lod_levels_) {
    total_memory += level.memory_usage;
  }

  return static_cast<float>(total_memory) / (1024.0f * 1024.0f);
}

template <typename R>
bool dynamic_lod_manager<R>::validate_configuration() const {
  std::lock_guard<std::mutex> lock(lod_mutex_);

  if (lod_levels_.empty())
    return false;

  // Check that we have at least one level for each complexity
  std::array<bool, 3> has_complexity = {false, false, false};
  for (const auto &level : lod_levels_) {
    if (static_cast<int>(level.complexity) >= 0 &&
        static_cast<int>(level.complexity) < 3) {
      has_complexity[static_cast<int>(level.complexity)] = true;
    }
  }

  return has_complexity[0] || has_complexity[1] || has_complexity[2];
}

template <typename R>
geometry_complexity_t
dynamic_lod_manager<R>::calculate_distance_based_lod(R distance) const {
  if (distance > config_.distance_threshold_coarse) {
    return COARSE;
  } else if (distance > config_.distance_threshold_medium) {
    return MEDIUM;
  } else {
    return FINE;
  }
}

template <typename R>
geometry_complexity_t dynamic_lod_manager<R>::calculate_performance_based_lod(
    float current_performance) const {
  if (current_performance > config_.performance_target_ms * 1.5f) {
    // Performance is poor, use coarser LOD
    return COARSE;
  } else if (current_performance > config_.performance_target_ms * 1.2f) {
    // Performance is okay but could be better
    return MEDIUM;
  } else {
    // Performance is good, use fine LOD
    return FINE;
  }
}

template <typename R>
void dynamic_lod_manager<R>::update_statistics(float frame_performance_ms) {
  stats_.total_frames++;
  stats_.current_performance_ms = frame_performance_ms;

  if (frame_performance_ms > 0) {
    // Update running average
    float alpha = 0.1f; // Smoothing factor
    if (stats_.avg_performance_ms == 0.0f) {
      stats_.avg_performance_ms = frame_performance_ms;
    } else {
      stats_.avg_performance_ms = alpha * frame_performance_ms +
                                  (1.0f - alpha) * stats_.avg_performance_ms;
    }
  }

  stats_.memory_usage_mb = get_memory_usage_mb();
}

template <typename R>
bool dynamic_lod_manager<R>::is_lod_transition_beneficial(
    geometry_complexity_t new_lod) const {
  // Find current and new LOD levels
  auto current_it = std::find_if(lod_levels_.begin(), lod_levels_.end(),
                                 [this](const lod_level<R> &level) {
                                   return level.complexity == current_lod_;
                                 });

  auto new_it = std::find_if(lod_levels_.begin(), lod_levels_.end(),
                             [new_lod](const lod_level<R> &level) {
                               return level.complexity == new_lod;
                             });

  if (current_it == lod_levels_.end() || new_it == lod_levels_.end()) {
    return false;
  }

  // Consider quality factor in decision
  float quality_threshold = config_.quality_factor;

  // If transitioning to coarser LOD, check if performance benefit outweighs
  // quality loss
  if (static_cast<int>(new_lod) < static_cast<int>(current_lod_)) {
    float performance_improvement =
        (current_it->performance_cost - new_it->performance_cost) /
        current_it->performance_cost;
    float quality_loss =
        (current_it->accuracy_estimate - new_it->accuracy_estimate) /
        current_it->accuracy_estimate;

    return performance_improvement > quality_loss * (1.0f - quality_threshold);
  }

  // If transitioning to finer LOD, check if we can afford the quality
  // improvement
  else if (static_cast<int>(new_lod) > static_cast<int>(current_lod_)) {
    float performance_cost =
        (new_it->performance_cost - current_it->performance_cost) /
        current_it->performance_cost;
    float quality_gain =
        (new_it->accuracy_estimate - current_it->accuracy_estimate) /
        current_it->accuracy_estimate;

    return performance_cost < quality_gain * quality_threshold;
  }

  return false;
}

// adaptive_lod_controller implementation
template <typename R>
adaptive_lod_controller<R>::adaptive_lod_controller(
    dynamic_lod_manager<R> &manager, size_t buffer_size,
    float performance_threshold, uint32_t adjustment_frequency)
    : lod_manager_(manager), buffer_size_(buffer_size),
      performance_threshold_(performance_threshold),
      adjustment_frequency_(adjustment_frequency), frame_counter_(0) {
  performance_buffer_.reserve(buffer_size);
}

template <typename R>
void adaptive_lod_controller<R>::update(const vec3<R> &view_position,
                                        const vec3<R> &object_position,
                                        float frame_performance_ms) {
  frame_counter_++;

  // Add performance to buffer
  performance_buffer_.push_back(frame_performance_ms);
  if (performance_buffer_.size() > buffer_size_) {
    performance_buffer_.erase(performance_buffer_.begin());
  }

  // Check if adjustment is needed
  if (needs_adjustment() && frame_counter_ % adjustment_frequency_ == 0) {
    float avg_performance = get_average_performance();
    float trend = get_performance_trend();

    // Predict future performance based on trend
    float predicted_performance =
        avg_performance + trend * adjustment_frequency_;

    lod_manager_.update_lod(view_position, object_position,
                            predicted_performance);
  }
}

template <typename R>
float adaptive_lod_controller<R>::get_average_performance() const {
  if (performance_buffer_.empty())
    return 0.0f;

  float sum = std::accumulate(performance_buffer_.begin(),
                              performance_buffer_.end(), 0.0f);
  return sum / performance_buffer_.size();
}

template <typename R>
float adaptive_lod_controller<R>::get_performance_trend() const {
  return calculate_trend();
}

template <typename R> void adaptive_lod_controller<R>::reset() {
  performance_buffer_.clear();
  frame_counter_ = 0;
}

template <typename R>
float adaptive_lod_controller<R>::calculate_trend() const {
  if (performance_buffer_.size() < 10)
    return 0.0f;

  // Simple linear regression to estimate trend
  size_t n = performance_buffer_.size();
  float sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;

  for (size_t i = 0; i < n; ++i) {
    float x = static_cast<float>(i);
    float y = performance_buffer_[i];
    sum_x += x;
    sum_y += y;
    sum_xy += x * y;
    sum_x2 += x * x;
  }

  float denominator = n * sum_x2 - sum_x * sum_x;
  if (std::abs(denominator) < 1e-6f)
    return 0.0f;

  return (n * sum_xy - sum_x * sum_y) / denominator;
}

template <typename R>
bool adaptive_lod_controller<R>::needs_adjustment() const {
  if (performance_buffer_.size() < buffer_size_ / 2)
    return false;

  float avg_performance = get_average_performance();
  float variance = 0.0f;

  for (float perf : performance_buffer_) {
    float diff = perf - avg_performance;
    variance += diff * diff;
  }

  variance /= performance_buffer_.size();
  float std_dev = std::sqrt(variance);

  // Need adjustment if performance is consistently poor or highly variable
  return (avg_performance > performance_threshold_ * 1.2f) ||
         (std_dev > performance_threshold_ * 0.3f);
}

// lod_geometry_cache implementation
template <typename R>
lod_geometry_cache<R>::lod_geometry_cache(size_t max_size, size_t max_memory_mb)
    : max_cache_size_(max_size), max_memory_mb_(max_memory_mb),
      current_frame_(0) {
  cache_entries_.reserve(max_size);
}

template <typename R>
geometry_interface<R> *lod_geometry_cache<R>::get_or_create_geometry(
    geometry_complexity_t complexity,
    std::function<std::unique_ptr<geometry_interface<R>>(geometry_complexity_t)>
        creator) {
  std::lock_guard<std::mutex> lock(cache_mutex_);

  cache_entry *entry = find_entry(complexity);
  if (entry) {
    update_access(*entry);
    return entry->geometry.get();
  }

  // Create new geometry
  auto geometry = creator(complexity);
  if (!geometry)
    return nullptr;

  entry = create_entry(complexity, std::move(geometry));
  if (!entry)
    return nullptr;

  // Evict LRU entries if needed
  evict_lru_entries();

  return entry->geometry.get();
}

template <typename R>
bool lod_geometry_cache<R>::prepare_geometry(geometry_complexity_t complexity) {
  std::lock_guard<std::mutex> lock(cache_mutex_);

  cache_entry *entry = find_entry(complexity);
  if (!entry || !entry->geometry)
    return false;

  if (!entry->is_prepared) {
    bool success = entry->geometry->prepare_for_gpu();
    if (success) {
      entry->is_prepared = true;
      entry->preparation_cost = entry->geometry->get_performance_cost();
    }
    return success;
  }

  return true;
}

template <typename R> void lod_geometry_cache<R>::evict_lru_entries() {
  size_t current_memory = calculate_memory_usage();
  if (current_memory <= max_memory_mb_ * 1024 * 1024 &&
      cache_entries_.size() <= max_cache_size_) {
    return; // No eviction needed
  }

  // Sort by last access time
  std::sort(cache_entries_.begin(), cache_entries_.end(),
            [](const cache_entry &a, const cache_entry &b) {
              return a.last_access_frame < b.last_access_frame;
            });

  // Evict entries until within limits
  auto it = cache_entries_.begin();
  while (it != cache_entries_.end() &&
         (cache_entries_.size() > max_cache_size_ ||
          calculate_memory_usage() > max_memory_mb_ * 1024 * 1024)) {
    it = cache_entries_.erase(it);
  }
}

template <typename R> void lod_geometry_cache<R>::clear_cache() {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  cache_entries_.clear();
}

template <typename R>
typename lod_geometry_cache<R>::cache_stats
lod_geometry_cache<R>::get_statistics() const {
  std::lock_guard<std::mutex> lock(cache_mutex_);

  cache_stats stats;
  stats.total_entries = cache_entries_.size();
  stats.prepared_entries = 0;
  stats.memory_usage_mb =
      static_cast<float>(calculate_memory_usage()) / (1024.0f * 1024.0f);

  uint32_t total_requests = 0;
  uint32_t cache_hits = 0;

  for (const auto &entry : cache_entries_) {
    if (entry.is_prepared)
      stats.prepared_entries++;
    total_requests += entry.access_count;
    if (entry.access_count > 1)
      cache_hits += entry.access_count - 1;
  }

  stats.total_requests = total_requests;
  stats.cache_hits = cache_hits;
  stats.hit_rate = total_requests > 0
                       ? static_cast<float>(cache_hits) / total_requests
                       : 0.0f;

  return stats;
}

template <typename R>
typename lod_geometry_cache<R>::cache_entry *
lod_geometry_cache<R>::find_entry(geometry_complexity_t complexity) {
  auto it = std::find_if(cache_entries_.begin(), cache_entries_.end(),
                         [complexity](const cache_entry &entry) {
                           return entry.complexity == complexity;
                         });

  return (it != cache_entries_.end()) ? &(*it) : nullptr;
}

template <typename R>
typename lod_geometry_cache<R>::cache_entry *
lod_geometry_cache<R>::create_entry(
    geometry_complexity_t complexity,
    std::unique_ptr<geometry_interface<R>> geometry) {
  if (cache_entries_.size() >= max_cache_size_) {
    evict_lru_entries();
  }

  if (cache_entries_.size() >= max_cache_size_) {
    return nullptr; // Still no space
  }

  cache_entry entry;
  entry.geometry = std::move(geometry);
  entry.complexity = complexity;
  entry.access_count = 1;
  entry.last_access_frame = current_frame_;
  entry.preparation_cost = 0.0f;
  entry.is_prepared = false;

  cache_entries_.push_back(std::move(entry));
  return &cache_entries_.back();
}

template <typename R>
void lod_geometry_cache<R>::update_access(cache_entry &entry) {
  entry.access_count++;
  entry.last_access_frame = current_frame_;
}

template <typename R>
size_t lod_geometry_cache<R>::calculate_memory_usage() const {
  size_t total = 0;
  for (const auto &entry : cache_entries_) {
    if (entry.geometry) {
      total += entry.geometry->estimate_memory_usage();
    }
  }
  return total;
}

// Utility functions implementation
namespace lod_utils {
template <typename R>
void calculate_optimal_lod_distances(
    const std::array<float, 3> &performance_costs,
    const std::array<float, 3> &accuracy_estimates, R target_performance_ms,
    std::array<R, 2> &distance_thresholds) {
  // Simple heuristic based on performance targets
  float coarse_cost = performance_costs[0];
  float medium_cost = performance_costs[1];
  float fine_cost = performance_costs[2];

  // Calculate distances where performance becomes unacceptable
  distance_thresholds[0] = 100.0 * (coarse_cost / target_performance_ms);
  distance_thresholds[1] = 100.0 * (medium_cost / target_performance_ms);

  // Ensure reasonable bounds
  distance_thresholds[0] =
      std::max(distance_thresholds[0], static_cast<R>(50.0));
  distance_thresholds[1] =
      std::max(distance_thresholds[1], static_cast<R>(200.0));
  distance_thresholds[1] =
      std::min(distance_thresholds[1], static_cast<R>(1000.0));
}

template <typename R>
float estimate_lod_performance(geometry_complexity_t complexity, R distance,
                               const lod_config<R> &config) {
  float base_cost = 1.0f;

  switch (complexity) {
  case COARSE:
    base_cost = 0.3f;
    break;
  case MEDIUM:
    base_cost = 1.0f;
    break;
  case FINE:
    base_cost = 2.5f;
    break;
  }

  // Adjust for distance (closer objects need more detail)
  float distance_factor = 1.0f;
  if (distance < config.distance_threshold_medium) {
    distance_factor = 2.0f;
  } else if (distance < config.distance_threshold_coarse) {
    distance_factor = 1.5f;
  }

  return base_cost * distance_factor;
}

template <typename R>
std::vector<std::vector<float>>
generate_lod_transition_matrix(const std::vector<lod_level<R>> &lod_levels) {
  size_t n = lod_levels.size();
  std::vector<std::vector<float>> matrix(n, std::vector<float>(n, 0.0f));

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      if (i == j) {
        matrix[i][j] = 1.0f; // Stay in same LOD
      } else {
        // Calculate transition cost based on performance difference
        float perf_diff = std::abs(lod_levels[i].performance_cost -
                                   lod_levels[j].performance_cost);
        float acc_diff = std::abs(lod_levels[i].accuracy_estimate -
                                  lod_levels[j].accuracy_estimate);
        matrix[i][j] = perf_diff / (perf_diff + acc_diff + 1e-6f);
      }
    }
  }

  return matrix;
}

} // namespace lod_utils

// Explicit template instantiations
template class dynamic_lod_manager<float>;
template class dynamic_lod_manager<double>;
template class adaptive_lod_controller<float>;
template class adaptive_lod_controller<double>;
template class lod_geometry_cache<float>;
template class lod_geometry_cache<double>;

} // namespace mqi
