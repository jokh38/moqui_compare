#ifndef MQI_PHASE4_BEAM_EYE_VIEW_HPP
#define MQI_PHASE4_BEAM_EYE_VIEW_HPP

/// \file
///
/// Phase 4: Beam-Eye View Projection System
///
/// This header implements beam-eye view projection for efficient 2D voxel mask
/// creation and coarse filtering in radiation therapy geometry.

#include "mqi_common.hpp"
#include "mqi_geometry_interface.hpp"
#include "mqi_vec.hpp"
#include <array>
#include <memory>
#include <vector>

namespace mqi {

/// Beam configuration for eye-view projection
template <typename R> struct beam_config {
  vec3<R> source_position;   ///< Beam source position
  vec3<R> beam_direction;    ///< Central beam axis (normalized)
  vec3<R> up_vector;         ///< Up vector for projection plane
  R field_size_x;            ///< Field size in X direction (mm)
  R field_size_y;            ///< Field size in Y direction (mm)
  R source_to_axis_distance; ///< Distance from source to isocenter (mm)
  R resolution_x;            ///< Resolution in X direction
  R resolution_y;            ///< Resolution in Y direction

  CUDA_HOST_DEVICE
  beam_config()
      : source_position(0, 0, -1000), beam_direction(0, 0, 1),
        up_vector(0, 1, 0), field_size_x(100), field_size_y(100),
        source_to_axis_distance(1000), resolution_x(1.0), resolution_y(1.0) {}
};

/// 2D projection result from beam-eye view
template <typename R> struct beam_projection {
  uint16_t width;                     ///< Projection width in pixels
  uint16_t height;                    ///< Projection height in pixels
  R pixel_size_x;                     ///< Pixel size in X (mm)
  R pixel_size_y;                     ///< Pixel size in Y (mm)
  vec3<R> projection_origin;          ///< World space origin of projection
  std::vector<float> intensity_map;   ///< Intensity/attenuation map
  std::vector<uint8_t> binary_mask;   ///< Binary occupancy mask
  std::vector<uint32_t> material_map; ///< Material ID per pixel
  R max_intensity;                    ///< Maximum intensity value

  beam_projection()
      : width(0), height(0), pixel_size_x(1.0), pixel_size_y(1.0),
        projection_origin(0), max_intensity(0.0) {}

  /// Resize projection arrays
  void resize(uint16_t w, uint16_t h) {
    width = w;
    height = h;
    intensity_map.resize(w * h, 0.0f);
    binary_mask.resize(w * h, 0);
    material_map.resize(w * h, 0);
  }

  /// Get intensity at pixel coordinates
  CUDA_HOST_DEVICE
  float get_intensity(uint16_t x, uint16_t y) const {
    if (x < width && y < height) {
      return intensity_map[y * width + x];
    }
    return 0.0f;
  }

  /// Set intensity at pixel coordinates
  CUDA_HOST_DEVICE
  void set_intensity(uint16_t x, uint16_t y, float intensity) {
    if (x < width && y < height) {
      intensity_map[y * width + x] = intensity;
      binary_mask[y * width + x] = (intensity > 0.0f) ? 1 : 0;
      max_intensity = std::max(max_intensity, static_cast<R>(intensity));
    }
  }

  /// Check if pixel is occupied
  CUDA_HOST_DEVICE
  bool is_occupied(uint16_t x, uint16_t y) const {
    if (x < width && y < height) {
      return binary_mask[y * width + x] != 0;
    }
    return false;
  }
};

/// Beam-eye view projector for creating 2D masks
template <typename R> class beam_eye_view_projector {
private:
  beam_config<R> config_;
  mat3x3<R> projection_matrix_;
  mat3x3<R> inverse_projection_matrix_;
  bool initialized_;

public:
  /// Constructor
  beam_eye_view_projector() : initialized_(false) {}

  /// Configure beam projection
  void configure(const beam_config<R> &config);

  /// Update projection matrices based on current configuration
  void update_projection_matrices();

  /// Project 3D geometry to 2D beam-eye view
  /// \param geometry Geometry to project
  /// \param resolution Output resolution
  /// \return 2D projection result
  beam_projection<R> project_geometry(const geometry_interface<R> *geometry,
                                      uint16_t resolution = 256);

  /// Project point from 3D to 2D coordinates
  CUDA_HOST_DEVICE
  bool project_point(const vec3<R> &world_point, vec2<R> &image_coords) const;

  /// Project ray from 2D image coordinates back to 3D
  CUDA_HOST_DEVICE
  bool unproject_point(const vec2<R> &image_coords, R depth,
                       vec3<R> &world_point) const;

  /// Check if 2D point corresponds to occupied geometry
  CUDA_HOST_DEVICE
  bool is_occupied_in_projection(const vec2<R> &image_coords,
                                 const beam_projection<R> &projection) const;

  /// Get beam configuration
  const beam_config<R> &get_config() const { return config_; }

  /// Set beam configuration
  void set_config(const beam_config<R> &config) {
    config_ = config;
    update_projection_matrices();
  }

private:
  /// Calculate projection matrix from beam configuration
  void calculate_projection_matrix();

  /// Perform ray casting for single pixel
  float cast_ray_for_pixel(const geometry_interface<R> *geometry,
                           const vec3<R> &ray_origin,
                           const vec3<R> &ray_direction, R max_distance) const;

  /// Sample geometry along ray for intensity calculation
  float sample_geometry_along_ray(const geometry_interface<R> *geometry,
                                  const vec3<R> &ray_origin,
                                  const vec3<R> &ray_direction, R max_distance,
                                  uint32_t num_samples = 100) const;
};

/// Adaptive beam projector that adjusts resolution based on geometry complexity
template <typename R> class adaptive_beam_projector {
private:
  beam_eye_view_projector<R> base_projector_;
  uint16_t base_resolution_;
  uint16_t max_resolution_;
  float complexity_threshold_;

public:
  /// Constructor
  adaptive_beam_projector(uint16_t base_res = 128, uint16_t max_res = 512,
                          float complexity_thresh = 0.1f)
      : base_resolution_(base_res), max_resolution_(max_res),
        complexity_threshold_(complexity_thresh) {}

  /// Project geometry with adaptive resolution
  beam_projection<R>
  project_geometry_adaptive(const geometry_interface<R> *geometry,
                            const beam_config<R> &beam_config);

  /// Estimate geometry complexity for adaptive resolution
  float
  estimate_geometry_complexity(const geometry_interface<R> *geometry) const;

  /// Set adaptive parameters
  void set_base_resolution(uint16_t res) { base_resolution_ = res; }
  void set_max_resolution(uint16_t res) { max_resolution_ = res; }
  void set_complexity_threshold(float thresh) {
    complexity_threshold_ = thresh;
  }
};

/// Multi-beam projector for handling multiple beam angles
template <typename R> class multi_beam_projector {
private:
  std::vector<beam_eye_view_projector<R>> beam_projectors_;
  std::vector<beam_config<R>> beam_configs_;
  std::vector<beam_projection<R>> projections_;

public:
  /// Add a beam configuration
  void add_beam(const beam_config<R> &config);

  /// Project geometry from all beam angles
  void project_geometry_all_beams(const geometry_interface<R> *geometry,
                                  uint16_t resolution = 256);

  /// Get projection for specific beam
  const beam_projection<R> &get_projection(size_t beam_index) const;

  /// Combine multiple projections into single mask
  beam_projection<R> combine_projections() const;

  /// Get number of beams
  size_t get_num_beams() const { return beam_projectors_.size(); }

  /// Clear all beams
  void clear() {
    beam_projectors_.clear();
    beam_configs_.clear();
    projections_.clear();
  }
};

/// Utility functions for beam-eye view operations
namespace beam_eye_view_utils {
/// Calculate optimal beam configuration for given geometry
template <typename R>
beam_config<R>
calculate_optimal_beam_config(const geometry_interface<R> *geometry,
                              const vec3<R> &target_point,
                              R source_distance = 1000.0);

/// Convert DICOM beam coordinates to local coordinates
template <typename R> vec3<R> dicom_to_local(const vec3<R> &dicom_coords);

/// Calculate beam divergence correction factor
template <typename R>
R calculate_divergence_correction(R distance_from_source, R reference_distance);

/// Generate beam profile for Gaussian or flat beams
template <typename R>
std::vector<float> generate_beam_profile(uint16_t resolution, R beam_size,
                                         bool gaussian = false,
                                         R sigma_factor = 2.0);

} // namespace beam_eye_view_utils

} // namespace mqi

#endif // MQI_PHASE4_BEAM_EYE_VIEW_HPP
