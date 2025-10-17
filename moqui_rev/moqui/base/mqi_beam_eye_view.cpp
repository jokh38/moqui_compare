#include "mqi_beam_eye_view.hpp"
#include "mqi_math.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace mqi {

// beam_eye_view_projector implementation
template <typename R>
void beam_eye_view_projector<R>::configure(const beam_config<R> &config) {
  config_ = config;
  update_projection_matrices();
  initialized_ = true;
}

template <typename R>
void beam_eye_view_projector<R>::update_projection_matrices() {
  calculate_projection_matrix();

  // Calculate inverse matrix for unprojection
  projection_matrix_ = projection_matrix_.inverse();
  inverse_projection_matrix_ = projection_matrix_.inverse();
}

template <typename R>
beam_projection<R> beam_eye_view_projector<R>::project_geometry(
    const geometry_interface<R> *geometry, uint16_t resolution) {
  if (!initialized_ || !geometry) {
    return beam_projection<R>();
  }

  beam_projection<R> projection;
  projection.resize(resolution, resolution);

  // Calculate pixel size based on field size and resolution
  projection.pixel_size_x = config_.field_size_x / resolution;
  projection.pixel_size_y = config_.field_size_y / resolution;

  // Set projection origin (center of field at isocenter plane)
  projection.projection_origin =
      config_.source_position +
      config_.beam_direction * config_.source_to_axis_distance -
      vec3<R>(config_.field_size_x / 2, config_.field_size_y / 2, 0);

  // Cast rays for each pixel
  for (uint16_t y = 0; y < resolution; ++y) {
    for (uint16_t x = 0; x < resolution; ++x) {
      // Calculate ray origin (pixel center in projection plane)
      vec3<R> pixel_center = projection.projection_origin +
                             vec3<R>((x + 0.5) * projection.pixel_size_x,
                                     (y + 0.5) * projection.pixel_size_y, 0);

      // Ray direction from source to pixel
      vec3<R> ray_direction = (pixel_center - config_.source_position);
      ray_direction.normalize();

      // Cast ray and accumulate intensity
      float intensity = sample_geometry_along_ray(
          geometry, config_.source_position, ray_direction,
          config_.source_to_axis_distance * 2.0);

      projection.set_intensity(x, y, intensity);

      // Get material at first intersection
      intersect_result_t<R> result =
          geometry->intersect(config_.source_position, ray_direction,
                              config_.source_to_axis_distance * 2.0);

      if (result.hit && x < resolution && y < resolution) {
        projection.material_map[y * resolution + x] = result.material_id;
      }
    }
  }

  return projection;
}

template <typename R>
bool beam_eye_view_projector<R>::project_point(const vec3<R> &world_point,
                                               vec2<R> &image_coords) const {
  if (!initialized_)
    return false;

  // Transform point to camera coordinate system
  vec3<R> relative_pos = world_point - config_.source_position;

  // Project onto image plane
  if (std::abs(relative_pos.dot(config_.beam_direction)) < 1e-6) {
    return false; // Point behind camera or too close
  }

  // Simple perspective projection
  R distance = relative_pos.dot(config_.beam_direction);
  vec3<R> projected = relative_pos - (config_.beam_direction * distance);

  // Convert to image coordinates
  image_coords.x =
      (projected.x + config_.field_size_x / 2) / config_.field_size_x;
  image_coords.y =
      (projected.y + config_.field_size_y / 2) / config_.field_size_y;

  return (image_coords.x >= 0 && image_coords.x <= 1 && image_coords.y >= 0 &&
          image_coords.y <= 1);
}

template <typename R>
bool beam_eye_view_projector<R>::unproject_point(const vec2<R> &image_coords,
                                                 R depth,
                                                 vec3<R> &world_point) const {
  if (!initialized_)
    return false;

  // Convert image coordinates to world coordinates on projection plane
  R world_x = (image_coords.x - 0.5) * config_.field_size_x;
  R world_y = (image_coords.y - 0.5) * config_.field_size_y;

  // Calculate point at given depth along beam
  world_point = config_.source_position + (config_.beam_direction * depth) +
                vec3<R>(world_x, world_y, 0);

  return true;
}

template <typename R>
bool beam_eye_view_projector<R>::is_occupied_in_projection(
    const vec2<R> &image_coords, const beam_projection<R> &projection) const {
  if (image_coords.x < 0 || image_coords.x >= 1 || image_coords.y < 0 ||
      image_coords.y >= 1) {
    return false;
  }

  uint16_t x = static_cast<uint16_t>(image_coords.x * projection.width);
  uint16_t y = static_cast<uint16_t>(image_coords.y * projection.height);

  return projection.is_occupied(x, y);
}

template <typename R>
void beam_eye_view_projector<R>::calculate_projection_matrix() {
  // Create orthonormal basis for projection
  vec3<R> right = (config_.beam_direction.cross(config_.up_vector));
  right.normalize();
  vec3<R> up = right.cross(config_.beam_direction);

  // Build projection matrix - Note: set_column is not available, we need to build the matrix differently
  projection_matrix_ = mat3x3<R>(right.x, up.x, config_.beam_direction.x,
                                right.y, up.y, config_.beam_direction.y,
                                right.z, up.z, config_.beam_direction.z);
}

template <typename R>
float beam_eye_view_projector<R>::cast_ray_for_pixel(
    const geometry_interface<R> *geometry, const vec3<R> &ray_origin,
    const vec3<R> &ray_direction, R max_distance) const {
  intersect_result_t<R> result =
      geometry->intersect(ray_origin, ray_direction, max_distance);

  if (result.hit) {
    return static_cast<float>(result.dist);
  }

  return 0.0f;
}

template <typename R>
float beam_eye_view_projector<R>::sample_geometry_along_ray(
    const geometry_interface<R> *geometry, const vec3<R> &ray_origin,
    const vec3<R> &ray_direction, R max_distance, uint32_t num_samples) const {
  float total_intensity = 0.0f;
  R step_size = max_distance / num_samples;

  for (uint32_t i = 0; i < num_samples; ++i) {
    R t = i * step_size;
    vec3<R> sample_point = ray_origin + ray_direction * t;

    if (geometry->is_inside(sample_point)) {
      R density = geometry->get_material_density(sample_point);
      total_intensity += static_cast<float>(density * step_size);
    }
  }

  return total_intensity;
}

// adaptive_beam_projector implementation
template <typename R>
beam_projection<R> adaptive_beam_projector<R>::project_geometry_adaptive(
    const geometry_interface<R> *geometry, const beam_config<R> &beam_config) {
  float complexity = estimate_geometry_complexity(geometry);

  // Choose resolution based on complexity
  uint16_t resolution = base_resolution_;
  if (complexity > complexity_threshold_) {
    float factor = std::min(complexity / complexity_threshold_, 4.0f);
    resolution = static_cast<uint16_t>(base_resolution_ * factor);
    resolution = std::min(resolution, max_resolution_);
  }

  base_projector_.configure(beam_config);
  return base_projector_.project_geometry(geometry, resolution);
}

template <typename R>
float adaptive_beam_projector<R>::estimate_geometry_complexity(
    const geometry_interface<R> *geometry) const {
  // Simple complexity estimation based on memory usage and performance cost
  size_t memory_usage = geometry->estimate_memory_usage();
  float performance_cost = geometry->get_performance_cost();

  // Normalize to 0-1 range (heuristic)
  float memory_factor =
      std::min(memory_usage / (1024.0f * 1024.0f), 1.0f); // MB to 0-1
  float performance_factor =
      std::min(performance_cost / 4.0f, 1.0f); // Cost to 0-1

  return (memory_factor + performance_factor) / 2.0f;
}

// multi_beam_projector implementation
template <typename R>
void multi_beam_projector<R>::add_beam(const beam_config<R> &config) {
  beam_configs_.push_back(config);
  beam_projectors_.emplace_back();
  beam_projectors_.back().configure(config);
}

template <typename R>
void multi_beam_projector<R>::project_geometry_all_beams(
    const geometry_interface<R> *geometry, uint16_t resolution) {
  projections_.clear();
  projections_.reserve(beam_projectors_.size());

  for (auto &projector : beam_projectors_) {
    projections_.push_back(projector.project_geometry(geometry, resolution));
  }
}

template <typename R>
const beam_projection<R> &
multi_beam_projector<R>::get_projection(size_t beam_index) const {
  if (beam_index < projections_.size()) {
    return projections_[beam_index];
  }

  static beam_projection<R> empty_projection;
  return empty_projection;
}

template <typename R>
beam_projection<R> multi_beam_projector<R>::combine_projections() const {
  if (projections_.empty()) {
    return beam_projection<R>();
  }

  beam_projection<R> combined = projections_[0];

  // Combine all projections using logical OR for binary mask
  for (size_t i = 1; i < projections_.size(); ++i) {
    const beam_projection<R> &current = projections_[i];

    if (current.width != combined.width || current.height != combined.height) {
      continue; // Skip incompatible projections
    }

    for (size_t j = 0; j < current.binary_mask.size(); ++j) {
      combined.binary_mask[j] =
          std::max(combined.binary_mask[j], current.binary_mask[j]);
      combined.intensity_map[j] += current.intensity_map[j];
    }
  }

  // Normalize intensity map
  float max_intensity = 0.0f;
  for (float intensity : combined.intensity_map) {
    max_intensity = std::max(max_intensity, intensity);
  }

  if (max_intensity > 0.0f) {
    for (float &intensity : combined.intensity_map) {
      intensity /= max_intensity;
    }
  }

  combined.max_intensity = 1.0f;
  return combined;
}

// Utility functions implementation
namespace beam_eye_view_utils {
template <typename R>
beam_config<R>
calculate_optimal_beam_config(const geometry_interface<R> *geometry,
                              const vec3<R> &target_point, R source_distance) {
  beam_config<R> config;

  // Get geometry bounds
  vec3<R> bounds_min, bounds_max;
  geometry->get_world_bounds(bounds_min, bounds_max);

  // Calculate field size to cover geometry
  R margin = 20.0; // 20mm margin
  config.field_size_x = (bounds_max.x - bounds_min.x) + 2 * margin;
  config.field_size_y = (bounds_max.y - bounds_min.y) + 2 * margin;

  // Position source
  config.source_position = target_point - vec3<R>(0, 0, source_distance);
  config.beam_direction = vec3<R>(0, 0, 1); // Assuming Z is beam direction
  config.up_vector = vec3<R>(0, 1, 0);
  config.source_to_axis_distance = source_distance;

  // Set resolution based on field size
  config.resolution_x = config.field_size_x / 256; // Aim for 256 pixels across
  config.resolution_y = config.field_size_y / 256;

  return config;
}

template <typename R> vec3<R> dicom_to_local(const vec3<R> &dicom_coords) {
  // Simple DICOM to local coordinate conversion
  // DICOM uses patient coordinate system (L,P,S) to (x,y,z)
  return vec3<R>(dicom_coords.x, -dicom_coords.y, dicom_coords.z);
}

template <typename R>
R calculate_divergence_correction(R distance_from_source,
                                  R reference_distance) {
  if (distance_from_source <= 0)
    return 1.0;
  return (reference_distance * reference_distance) /
         (distance_from_source * distance_from_source);
}

template <typename R>
std::vector<float> generate_beam_profile(uint16_t resolution, R beam_size,
                                         bool gaussian, R sigma_factor) {
  std::vector<float> profile(resolution * resolution);
  R center = resolution / 2.0;
  R beam_radius = beam_size / 2.0;

  for (uint16_t y = 0; y < resolution; ++y) {
    for (uint16_t x = 0; x < resolution; ++x) {
      R dx = (x - center) * (beam_size / resolution);
      R dy = (y - center) * (beam_size / resolution);
      R r = std::sqrt(dx * dx + dy * dy);

      float intensity = 0.0f;
      if (r <= beam_radius) {
        if (gaussian) {
          R sigma = beam_radius / sigma_factor;
          intensity = std::exp(-(r * r) / (2.0 * sigma * sigma));
        } else {
          intensity = 1.0f; // Flat beam
        }
      }

      profile[y * resolution + x] = intensity;
    }
  }

  return profile;
}
} // namespace beam_eye_view_utils

// Explicit template instantiations
template class beam_eye_view_projector<float>;
template class beam_eye_view_projector<double>;
template class adaptive_beam_projector<float>;
template class adaptive_beam_projector<double>;
template class multi_beam_projector<float>;
template class multi_beam_projector<double>;
template struct beam_config<float>;
template struct beam_config<double>;
template struct beam_projection<float>;
template struct beam_projection<double>;

} // namespace mqi
