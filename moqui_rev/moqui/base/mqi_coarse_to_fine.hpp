#ifndef MQI_PHASE3_COARSE_TO_FINE_HPP
#define MQI_PHASE3_COARSE_TO_FINE_HPP

/// \file
/// \brief Phase 3.0: Coarse-to-Fine Acceleration Framework for GPU Optimization
///
/// This header implements a hierarchical filtering system to optimize particle
/// transport simulation by reducing unnecessary calculations through early
/// exclusion and progressive refinement. The framework follows three levels of
/// inspection:
///
/// Level 1: Bounding box inspection (AABB test)
/// Level 2: Simplified geometry approximation
/// Level 3: Detailed calculation pipeline
///
/// Features:
/// - Early exit optimizations based on energy thresholds
/// - Statistical importance sampling
/// - Depth-based termination
/// - Warp-level consistency optimizations
/// - GPU-friendly memory access patterns

#include "mqi_common.hpp"
#include "mqi_geometry.hpp"
#include "mqi_grid3d.hpp"
#include "mqi_material.hpp"
#include "mqi_math.hpp"
#include "mqi_track.hpp"
#include "mqi_vec.hpp"
#include <cmath>
#include <cstdint>
#include <type_traits> // For alignas

namespace mqi {

// ============================================================================
// CONSTANTS AND CONFIGURATION
// ============================================================================

/// Maximum number of bounding boxes that can be stored
constexpr uint32_t MAX_BOUNDING_BOXES = 1024;

/// Maximum number of simplified geometry objects
constexpr uint32_t MAX_SIMPLIFIED_GEOMETRIES = 256;

/// Energy threshold for early termination (MeV)
constexpr float EARLY_EXIT_ENERGY_THRESHOLD = 0.1f;

/// Statistical importance threshold for early termination
constexpr float STATISTICAL_IMPORTANCE_THRESHOLD = 1e-6f;

/// Maximum depth before forced termination
constexpr uint32_t MAX_DEPTH_THRESHOLD = 1000;

/// Warp size for CUDA (threads per warp)
constexpr uint32_t WARP_SIZE = 32;

// ============================================================================
// ENUMERATIONS
// ============================================================================

/// Coarse-to-fine inspection levels
typedef enum {
  COARSE_LEVEL_1_BOUNDING_BOX = 1,   ///< AABB bounding box test
  COARSE_LEVEL_2_SIMPLIFIED_GEO = 2, ///< Simplified geometry approximation
  COARSE_LEVEL_3_DETAILED_CALC = 3   ///< Detailed physics calculation
} inspection_level_t;

/// Early exit reasons
typedef enum {
  EXIT_NONE = 0,             ///< No early exit
  EXIT_ENERGY_THRESHOLD = 1, ///< Energy below threshold
  EXIT_STATISTICAL = 2,      ///< Statistical importance too low
  EXIT_DEPTH_LIMIT = 3,      /// Maximum depth reached
  EXIT_BOUNDARY = 4,         ///< Exited simulation boundary
  EXIT_CUSTOM = 5            ///< User-defined criteria
} early_exit_reason_t;

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/// Axis-Aligned Bounding Box (AABB) for Level 1 filtering
#ifdef __CUDACC__
struct __align__(16) bounding_box_t
#else
struct alignas(16) bounding_box_t
#endif
{
  vec3<float> min_point;   ///< Minimum corner of the box
  vec3<float> max_point;   ///< Maximum corner of the box
  uint32_t geometry_id;    ///< Associated geometry identifier
  uint32_t material_id;    ///< Material identifier
  float importance_weight; ///< Statistical importance weight
  uint32_t padding[2];     ///< Alignment padding

  /// Default constructor
  CUDA_HOST_DEVICE
  bounding_box_t() : geometry_id(0), material_id(0), importance_weight(1.0f) {
    min_point = vec3<float>(0.0f, 0.0f, 0.0f);
    max_point = vec3<float>(0.0f, 0.0f, 0.0f);
  }

  /// Constructor with bounds
  CUDA_HOST_DEVICE
  bounding_box_t(const vec3<float> &min_pt, const vec3<float> &max_pt,
                 uint32_t geo_id = 0, uint32_t mat_id = 0,
                 float importance = 1.0f)
      : min_point(min_pt), max_point(max_pt), geometry_id(geo_id),
        material_id(mat_id), importance_weight(importance) {
    ;
  }

  /// Check if a point is inside the bounding box
  CUDA_HOST_DEVICE
  bool contains_point(const vec3<float> &point) const {
    return (point.x >= min_point.x && point.x <= max_point.x &&
            point.y >= min_point.y && point.y <= max_point.y &&
            point.z >= min_point.z && point.z <= max_point.z);
  }

  /// Check if a ray intersects the bounding box
  CUDA_HOST_DEVICE
  bool ray_intersects(const vec3<float> &ray_origin,
                      const vec3<float> &ray_direction, float &t_near,
                      float &t_far) const {
    vec3<float> inv_dir(1.0f / ray_direction.x, 1.0f / ray_direction.y,
                        1.0f / ray_direction.z);

    vec3<float> diff0 = min_point - ray_origin;
    vec3<float> diff1 = max_point - ray_origin;
    vec3<float> t0(diff0.x * inv_dir.x, diff0.y * inv_dir.y,
                   diff0.z * inv_dir.z);
    vec3<float> t1(diff1.x * inv_dir.x, diff1.y * inv_dir.y,
                   diff1.z * inv_dir.z);

    vec3<float> t_min =
        vec3<float>(fminf(t0.x, t1.x), fminf(t0.y, t1.y), fminf(t0.z, t1.z));
    vec3<float> t_max =
        vec3<float>(fmaxf(t0.x, t1.x), fmaxf(t0.y, t1.y), fmaxf(t0.z, t1.z));

    t_near = fmaxf(fmaxf(t_min.x, t_min.y), t_min.z);
    t_far = fminf(fminf(t_max.x, t_max.y), t_max.z);

    return t_far >= t_near && t_far >= 0.0f;
  }

  /// Get the center point of the bounding box
  CUDA_HOST_DEVICE
  vec3<float> get_center() const { return (min_point + max_point) * 0.5f; }

  /// Get the dimensions of the bounding box
  CUDA_HOST_DEVICE
  vec3<float> get_dimensions() const { return max_point - min_point; }

  /// Get the volume of the bounding box
  CUDA_HOST_DEVICE
  float get_volume() const {
    vec3<float> dims = get_dimensions();
    return dims.x * dims.y * dims.z;
  }
};

/// Simplified geometry object for Level 2 filtering
#ifdef __CUDACC__
struct __align__(32) simplified_geometry_t
#else
struct alignas(32) simplified_geometry_t
#endif
{
  vec3<float> center;     ///< Center position
  vec3<float> dimensions; ///< Dimensions (for box-like approximation)
  float radius;           ///< Radius (for sphere-like approximation)
  uint32_t geometry_type; ///< Type of simplified geometry (0=box, 1=sphere,
                          ///< 2=cylinder)
  uint32_t parent_bounding_box; ///< Parent bounding box ID
  float density_factor;         ///< Relative density factor
  uint32_t padding[3];          ///< Alignment padding

  /// Default constructor
  CUDA_HOST_DEVICE
  simplified_geometry_t()
      : radius(0.0f), geometry_type(0), parent_bounding_box(0),
        density_factor(1.0f) {
    center = vec3<float>(0.0f, 0.0f, 0.0f);
    dimensions = vec3<float>(0.0f, 0.0f, 0.0f);
  }

  /// Constructor for box-like geometry
  CUDA_HOST_DEVICE
  simplified_geometry_t(const vec3<float> &ctr, const vec3<float> &dims,
                        uint32_t bbox_id = 0, float density = 1.0f)
      : center(ctr), dimensions(dims), radius(0.0f), geometry_type(0),
        parent_bounding_box(bbox_id), density_factor(density) {
    ;
  }

  /// Constructor for sphere-like geometry
  CUDA_HOST_DEVICE
  simplified_geometry_t(const vec3<float> &ctr, float rad, uint32_t bbox_id = 0,
                        float density = 1.0f)
      : center(ctr), dimensions(0.0f, 0.0f, 0.0f), radius(rad),
        geometry_type(1), parent_bounding_box(bbox_id),
        density_factor(density) {
    ;
  }

  /// Check if a point is inside the simplified geometry
  CUDA_HOST_DEVICE
  bool contains_point(const vec3<float> &point) const {
    if (geometry_type == 0) { // Box
      vec3<float> half_dims = dimensions * 0.5f;
      vec3<float> rel_pos = point - center;
      return (fabsf(rel_pos.x) <= half_dims.x &&
              fabsf(rel_pos.y) <= half_dims.y &&
              fabsf(rel_pos.z) <= half_dims.z);
    } else if (geometry_type == 1) { // Sphere
      vec3<float> rel_pos = point - center;
      return ((rel_pos.x * rel_pos.x + rel_pos.y * rel_pos.y +
               rel_pos.z * rel_pos.z) <= radius * radius);
    }
    return false;
  }

  /// Approximate ray intersection distance
  CUDA_HOST_DEVICE
  float approximate_ray_intersection(const vec3<float> &ray_origin,
                                     const vec3<float> &ray_direction) const {
    if (geometry_type == 0) { // Box - use center distance as approximation
      vec3<float> rel_pos = center - ray_origin;
      float projection = rel_pos.dot(ray_direction);
      if (projection > 0) {
        return projection;
      }
    } else if (geometry_type == 1) { // Sphere - exact solution
      vec3<float> oc = ray_origin - center;
      float b = 2.0f * oc.dot(ray_direction);
      float c = (oc.x * oc.x + oc.y * oc.y + oc.z * oc.z) - radius * radius;
      float discriminant = b * b - 4.0f * c;
      if (discriminant >= 0) {
        return (-b - sqrtf(discriminant)) * 0.5f;
      }
    }
    return -1.0f; // No intersection
  }
};

/// Particle state for coarse-to-fine filtering
#ifdef __CUDACC__
struct __align__(64) particle_filter_state_t
#else
struct alignas(64) particle_filter_state_t
#endif
{
  vec3<float> position;            ///< Current particle position
  vec3<float> direction;           ///< Current particle direction
  float energy;                    ///< Current particle energy (MeV)
  float statistical_weight;        ///< Statistical weight of particle
  uint32_t depth;                  ///< Current interaction depth
  uint32_t current_level;          ///< Current inspection level
  early_exit_reason_t exit_reason; ///< Reason for early exit
  bool active;                     ///< Whether particle is still active
  uint32_t padding[4];             ///< Alignment padding

  /// Default constructor
  CUDA_HOST_DEVICE
  particle_filter_state_t()
      : energy(0.0f), statistical_weight(1.0f), depth(0),
        current_level(COARSE_LEVEL_1_BOUNDING_BOX), exit_reason(EXIT_NONE),
        active(false) {
    position = vec3<float>(0.0f, 0.0f, 0.0f);
    direction = vec3<float>(0.0f, 0.0f, 1.0f);
  }

  /// Initialize from track template
  template <typename R>
  CUDA_HOST_DEVICE void initialize_from_track(const track_t<R> &track) {
    position.x = static_cast<float>(track.vtx1.pos.x);
    position.y = static_cast<float>(track.vtx1.pos.y);
    position.z = static_cast<float>(track.vtx1.pos.z);
    direction.x = static_cast<float>(track.vtx1.dir.x);
    direction.y = static_cast<float>(track.vtx1.dir.y);
    direction.z = static_cast<float>(track.vtx1.dir.z);
    energy = static_cast<float>(track.vtx1.ke);
    statistical_weight = 1.0f; // TODO: get from track if available
    depth = 0;
    current_level = COARSE_LEVEL_1_BOUNDING_BOX;
    exit_reason = EXIT_NONE;
    active = true;
  }

  /// Check if particle should be terminated early
  CUDA_HOST_DEVICE
  bool should_terminate_early() const {
    if (energy < EARLY_EXIT_ENERGY_THRESHOLD) {
      return true;
    }
    if (statistical_weight < STATISTICAL_IMPORTANCE_THRESHOLD) {
      return true;
    }
    if (depth > MAX_DEPTH_THRESHOLD) {
      return true;
    }
    return false;
  }

  /// Get early exit reason
  CUDA_HOST_DEVICE
  early_exit_reason_t get_exit_reason() const {
    if (energy < EARLY_EXIT_ENERGY_THRESHOLD) {
      return EXIT_ENERGY_THRESHOLD;
    }
    if (statistical_weight < STATISTICAL_IMPORTANCE_THRESHOLD) {
      return EXIT_STATISTICAL;
    }
    if (depth > MAX_DEPTH_THRESHOLD) {
      return EXIT_DEPTH_LIMIT;
    }
    return EXIT_NONE;
  }
};

/// Warp-level filtering state for optimizing branch divergence
#ifdef __CUDACC__
struct __align__(256) warp_filter_state_t
#else
struct alignas(256) warp_filter_state_t
#endif
{
  uint32_t active_mask;           ///< Bitmask of active threads in warp
  uint32_t level_counts[4];       ///< Count of threads at each inspection level
  uint32_t exit_reason_counts[6]; ///< Count of threads for each exit reason
  float total_energy;             ///< Total energy of particles in warp
  float total_statistical_weight; ///< Total statistical weight in warp
  uint32_t padding[8];            ///< Alignment padding

  /// Default constructor
  CUDA_HOST_DEVICE
  warp_filter_state_t()
      : active_mask(0xFFFFFFFF), total_energy(0.0f),
        total_statistical_weight(0.0f) {
    for (int i = 0; i < 4; ++i)
      level_counts[i] = 0;
    for (int i = 0; i < 6; ++i)
      exit_reason_counts[i] = 0;
  }

  /// Initialize for a warp
  CUDA_HOST_DEVICE
  void initialize_warp() {
    active_mask = 0xFFFFFFFF;
    total_energy = 0.0f;
    total_statistical_weight = 0.0f;
    for (int i = 0; i < 4; ++i)
      level_counts[i] = 0;
    for (int i = 0; i < 6; ++i)
      exit_reason_counts[i] = 0;
  }

  /// Update statistics for a thread
  CUDA_HOST_DEVICE
  void update_thread_stats(uint32_t lane_id,
                           const particle_filter_state_t &particle) {
    if (particle.active) {
      total_energy += particle.energy;
      total_statistical_weight += particle.statistical_weight;
      level_counts[particle.current_level]++;
    } else {
      exit_reason_counts[particle.exit_reason]++;
      // Deactivate thread bit
      active_mask &= ~(1u << lane_id);
    }
  }

  /// Check if warp has significant activity
  CUDA_HOST_DEVICE
  bool has_significant_activity() const {
    return (total_energy > EARLY_EXIT_ENERGY_THRESHOLD * WARP_SIZE) ||
           (total_statistical_weight >
            STATISTICAL_IMPORTANCE_THRESHOLD * WARP_SIZE);
  }

  /// Get the dominant inspection level for this warp
  CUDA_HOST_DEVICE
  inspection_level_t get_dominant_level() const {
    uint32_t max_count = 0;
    inspection_level_t dominant = COARSE_LEVEL_1_BOUNDING_BOX;

    for (int i = 1; i <= 3; ++i) {
      if (level_counts[i] > max_count) {
        max_count = level_counts[i];
        dominant = static_cast<inspection_level_t>(i);
      }
    }
    return dominant;
  }
};

// ============================================================================
// COARSE-TO-FINE FRAMEWORK API
// ============================================================================

/// Initialize Phase 3.0 Coarse-to-Fine framework
/**
 * \param bounding_boxes Array of bounding boxes to initialize
 * \param num_bboxes Number of bounding boxes
 * \param simplified_geometries Array of simplified geometries
 * \param num_geometries Number of simplified geometries
 * \return true if initialization successful, false otherwise
 */
bool initialize_coarse_to_fine_framework(
    bounding_box_t *bounding_boxes, uint32_t num_bboxes,
    simplified_geometry_t *simplified_geometries, uint32_t num_geometries);

/// Shutdown Phase 3.0 framework
void shutdown_coarse_to_fine_framework();

/// Level 1: Bounding box inspection test
/**
 * \param particle Particle filter state to test
 * \param bounding_boxes Array of bounding boxes
 * \param num_bboxes Number of bounding boxes
 * \return true if particle passes Level 1 test, false otherwise
 */
CUDA_HOST_DEVICE
bool level1_bounding_box_test(particle_filter_state_t &particle,
                              const bounding_box_t *bounding_boxes,
                              uint32_t num_bboxes);

/// Level 2: Simplified geometry approximation test
/**
 * \param particle Particle filter state to test
 * \param simplified_geometries Array of simplified geometries
 * \param num_geometries Number of simplified geometries
 * \return true if particle passes Level 2 test, false otherwise
 */
CUDA_HOST_DEVICE
bool level2_simplified_geometry_test(
    particle_filter_state_t &particle,
    const simplified_geometry_t *simplified_geometries,
    uint32_t num_geometries);

/// Level 3: Detailed calculation pipeline entry point
/**
 * \param particle Particle filter state to process
 * \param track Original particle track for detailed calculations
 * \return true if detailed calculation should proceed, false if terminated
 */
template <typename R>
CUDA_HOST_DEVICE bool
level3_detailed_calculation(particle_filter_state_t &particle,
                            track_t<R> &track);

/// Main coarse-to-fine filtering pipeline
/**
 * \param particles Array of particle filter states
 * \param tracks Array of original particle tracks
 * \param num_particles Number of particles to process
 * \param warp_state Warp-level state for optimization
 * \param bounding_boxes Array of bounding boxes
 * \param num_bboxes Number of bounding boxes
 * \param simplified_geometries Array of simplified geometries
 * \param num_geometries Number of simplified geometries
 */
template <typename R>
CUDA_DEVICE void coarse_to_fine_filter_pipeline(
    particle_filter_state_t *particles, track_t<R> *tracks,
    uint32_t num_particles, warp_filter_state_t &warp_state,
    const bounding_box_t *bounding_boxes, uint32_t num_bboxes,
    const simplified_geometry_t *simplified_geometries,
    uint32_t num_geometries);

/// Warp-level consistency optimization
/**
 * \param particles Array of particle filter states in warp
 * \param warp_state Warp-level state
 * \param lane_id Thread lane ID within warp
 */
CUDA_DEVICE
void optimize_warp_consistency(particle_filter_state_t *particles,
                               warp_filter_state_t &warp_state,
                               uint32_t lane_id);

/// Early exit evaluation
/**
 * \param particle Particle filter state to evaluate
 * \return true if particle should exit early, false otherwise
 */
CUDA_HOST_DEVICE
bool evaluate_early_exit(particle_filter_state_t &particle);

/// Create bounding box from geometry
/**
 * \param geometry Source geometry object
 * \param padding Optional padding around geometry
 * \return Generated bounding box
 */
bounding_box_t create_bounding_box_from_geometry(const geometry &geometry,
                                                 float padding = 0.0f);

/// Create simplified geometry from detailed geometry
/**
 * \param geometry Source detailed geometry
 * \param simplification_level Level of simplification (1-10, higher = more
 * simplified) \return Generated simplified geometry
 */
simplified_geometry_t
create_simplified_geometry(const geometry &geometry,
                           uint32_t simplification_level = 5);

// ============================================================================
// STATISTICS AND MONITORING
// ============================================================================

/// Coarse-to-fine performance statistics
struct coarse_to_fine_stats_t {
  uint64_t total_particles_processed; ///< Total particles processed
  uint64_t level1_passes;             ///< Particles passing Level 1
  uint64_t level2_passes;             ///< Particles passing Level 2
  uint64_t level3_processed;          ///< Particles reaching Level 3
  uint64_t early_exits[6];            ///< Early exits by reason
  float average_processing_time_us;   ///< Average processing time per particle
  float level_hit_ratios[4];          ///< Hit ratios for each level
  uint32_t padding[8];                ///< Alignment padding

  /// Default constructor
  coarse_to_fine_stats_t()
      : total_particles_processed(0), level1_passes(0), level2_passes(0),
        level3_processed(0), average_processing_time_us(0.0f) {
    for (int i = 0; i < 6; ++i)
      early_exits[i] = 0;
    for (int i = 0; i < 4; ++i)
      level_hit_ratios[i] = 0.0f;
  }

  /// Calculate hit ratios
  void calculate_hit_ratios() {
    if (total_particles_processed > 0) {
      level_hit_ratios[1] =
          static_cast<float>(level1_passes) / total_particles_processed;
      level_hit_ratios[2] =
          static_cast<float>(level2_passes) / total_particles_processed;
      level_hit_ratios[3] =
          static_cast<float>(level3_processed) / total_particles_processed;
    }
  }

  /// Reset statistics
  void reset() {
    total_particles_processed = 0;
    level1_passes = 0;
    level2_passes = 0;
    level3_processed = 0;
    average_processing_time_us = 0.0f;
    for (int i = 0; i < 6; ++i)
      early_exits[i] = 0;
    for (int i = 0; i < 4; ++i)
      level_hit_ratios[i] = 0.0f;
  }
};

/// Get coarse-to-fine performance statistics
/**
 * \return Current performance statistics
 */
coarse_to_fine_stats_t get_coarse_to_fine_statistics();

/// Reset coarse-to-fine performance statistics
void reset_coarse_to_fine_statistics();

/// Print performance summary
void print_coarse_to_fine_performance_summary();

} // namespace mqi

#endif // MQI_PHASE3_COARSE_TO_FINE_HPP
