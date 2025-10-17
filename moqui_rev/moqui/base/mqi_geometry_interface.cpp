#include "mqi_geometry_interface.hpp"
#include <algorithm>
#include <stdexcept>

namespace mqi {

// Geometry factory implementation
template <typename R>
std::unique_ptr<geometry_interface<R>>
geometry_factory<R>::create_geometry(geometry_type type,
                                     geometry_precision_t precision,
                                     geometry_complexity_t complexity) {
  // This will be implemented when we create specific geometry types
  // For now, return nullptr to indicate not implemented
  return nullptr;
}

template <typename R>
void geometry_factory<R>::register_geometry_type(
    geometry_type type,
    std::unique_ptr<geometry_interface<R>> (*creator)(geometry_precision_t,
                                                      geometry_complexity_t)) {
  // This will be implemented with a registry system
  // For now, this is a placeholder
}

template <typename R>
std::vector<geometry_type> geometry_factory<R>::get_available_types() {
  // Return empty for now, will be populated as we implement geometry types
  return std::vector<geometry_type>();
}

template <typename R>
bool geometry_factory<R>::is_type_supported(geometry_type type) {
  // Will check against registry once implemented
  return false;
}

// Geometry manager implementation
template <typename R>
void geometry_manager<R>::add_geometry(
    std::unique_ptr<geometry_interface<R>> geometry) {
  if (geometry) {
    geometries_.push_back(std::move(geometry));
  }
}

template <typename R> void geometry_manager<R>::remove_geometry(size_t index) {
  if (index < geometries_.size()) {
    geometries_.erase(geometries_.begin() + index);
  }
}

template <typename R>
geometry_interface<R> *geometry_manager<R>::get_geometry(size_t index) {
  if (index < geometries_.size()) {
    return geometries_[index].get();
  }
  return nullptr;
}

template <typename R>
intersect_result_t<R> geometry_manager<R>::find_nearest_intersection(
    const vec3<R> &origin, const vec3<R> &direction, R max_distance) const {
  intersect_result_t<R> nearest_result;
  R nearest_distance = max_distance;

  for (const auto &geometry : geometries_) {
    intersect_result_t<R> result =
        geometry->intersect(origin, direction, max_distance);

    if (result.hit && result.dist >= 0 && result.dist < nearest_distance) {
      nearest_result = result;
      nearest_distance = result.dist;
    }
  }

  return nearest_result;
}

template <typename R>
void geometry_manager<R>::set_global_precision(geometry_precision_t precision) {
  global_precision_ = precision;
  for (auto &geometry : geometries_) {
    geometry->set_precision(precision);
  }
}

template <typename R>
void geometry_manager<R>::set_global_complexity(
    geometry_complexity_t complexity) {
  global_complexity_ = complexity;
  for (auto &geometry : geometries_) {
    geometry->set_complexity(complexity);
  }
}

template <typename R> bool geometry_manager<R>::prepare_all_for_gpu() {
  bool all_success = true;
  for (auto &geometry : geometries_) {
    if (!geometry->prepare_for_gpu()) {
      all_success = false;
    }
  }
  return all_success;
}

template <typename R> void geometry_manager<R>::cleanup_all_gpu() {
  for (auto &geometry : geometries_) {
    geometry->cleanup_gpu();
  }
}

template <typename R>
size_t geometry_manager<R>::get_total_memory_usage() const {
  size_t total = 0;
  for (const auto &geometry : geometries_) {
    total += geometry->estimate_memory_usage();
  }
  return total;
}

template <typename R> bool geometry_manager<R>::validate_all() const {
  for (const auto &geometry : geometries_) {
    if (!geometry->validate()) {
      return false;
    }
  }
  return true;
}

// Explicit template instantiations
template class geometry_factory<float>;
template class geometry_factory<double>;
template class geometry_manager<float>;
template class geometry_manager<double>;

} // namespace mqi
