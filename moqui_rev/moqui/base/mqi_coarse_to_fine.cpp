#include "mqi_coarse_to_fine.hpp"
#include <chrono>
#include <cstring>
#include <iostream>

namespace mqi {

// ============================================================================
// GLOBAL STATE AND STATISTICS
// ============================================================================

/// Global bounding boxes array
static bounding_box_t *g_bounding_boxes = nullptr;
static uint32_t g_num_bounding_boxes = 0;

/// Global simplified geometries array
static simplified_geometry_t *g_simplified_geometries = nullptr;
static uint32_t g_num_simplified_geometries = 0;

/// Global performance statistics
static coarse_to_fine_stats_t g_performance_stats;

// ============================================================================
// FRAMEWORK INITIALIZATION AND SHUTDOWN
// ============================================================================

bool initialize_coarse_to_fine_framework(
    bounding_box_t *bounding_boxes, uint32_t num_bboxes,
    simplified_geometry_t *simplified_geometries, uint32_t num_geometries) {
  // Clear any existing state
  shutdown_coarse_to_fine_framework();

  // Validate inputs
  if (num_bboxes > MAX_BOUNDING_BOXES ||
      num_geometries > MAX_SIMPLIFIED_GEOMETRIES) {
    std::cerr << "Error: Exceeded maximum number of geometries" << std::endl;
    return false;
  }

  if (!bounding_boxes && num_bboxes > 0) {
    std::cerr << "Error: Null bounding boxes array with non-zero count"
              << std::endl;
    return false;
  }

  if (!simplified_geometries && num_geometries > 0) {
    std::cerr << "Error: Null simplified geometries array with non-zero count"
              << std::endl;
    return false;
  }

  // Allocate and copy bounding boxes
  if (num_bboxes > 0) {
    g_bounding_boxes = new bounding_box_t[num_bboxes];
    memcpy(g_bounding_boxes, bounding_boxes,
           num_bboxes * sizeof(bounding_box_t));
    g_num_bounding_boxes = num_bboxes;
  }

  // Allocate and copy simplified geometries
  if (num_geometries > 0) {
    g_simplified_geometries = new simplified_geometry_t[num_geometries];
    memcpy(g_simplified_geometries, simplified_geometries,
           num_geometries * sizeof(simplified_geometry_t));
    g_num_simplified_geometries = num_geometries;
  }

  // Reset statistics
  g_performance_stats.reset();

  std::cout << "Coarse-to-Fine framework initialized successfully" << std::endl;
  std::cout << "  Bounding boxes: " << g_num_bounding_boxes << std::endl;
  std::cout << "  Simplified geometries: " << g_num_simplified_geometries
            << std::endl;

  return true;
}

void shutdown_coarse_to_fine_framework() {
  // Clean up bounding boxes
  if (g_bounding_boxes) {
    delete[] g_bounding_boxes;
    g_bounding_boxes = nullptr;
  }
  g_num_bounding_boxes = 0;

  // Clean up simplified geometries
  if (g_simplified_geometries) {
    delete[] g_simplified_geometries;
    g_simplified_geometries = nullptr;
  }
  g_num_simplified_geometries = 0;

  // Reset statistics
  g_performance_stats.reset();

  std::cout << "Coarse-to-Fine framework shutdown complete" << std::endl;
}

// ============================================================================
// LEVEL 1: BOUNDING BOX INSPECTION (AABB TEST)
// ============================================================================

CUDA_HOST_DEVICE
bool level1_bounding_box_test(particle_filter_state_t &particle,
                              const bounding_box_t *bounding_boxes,
                              uint32_t num_bboxes) {
  if (num_bboxes == 0 || !bounding_boxes) {
    return true; // No bounding boxes to test against, pass by default
  }

  // Check if particle already has a target bounding box
  uint32_t current_bbox_id = 0; // TODO: Get from particle state if available

  // Test against current bounding box first
  if (current_bbox_id < num_bboxes) {
    const bounding_box_t &bbox = bounding_boxes[current_bbox_id];

    // Quick point-in-bbox test
    if (bbox.contains_point(particle.position)) {
      particle.current_level = COARSE_LEVEL_2_SIMPLIFIED_GEO;
      return true;
    }

    // Ray intersection test for future movement
    float t_near, t_far;
    if (bbox.ray_intersects(particle.position, particle.direction, t_near,
                            t_far)) {
      // Check if intersection is within reasonable distance
      if (t_near < 10.0f) { // 10 cm look-ahead distance
        particle.current_level = COARSE_LEVEL_2_SIMPLIFIED_GEO;
        return true;
      }
    }
  }

  // Search all bounding boxes if particle doesn't have a current target
  for (uint32_t i = 0; i < num_bboxes; ++i) {
    const bounding_box_t &bbox = bounding_boxes[i];

    // Point-in-bbox test
    if (bbox.contains_point(particle.position)) {
      particle.current_level = COARSE_LEVEL_2_SIMPLIFIED_GEO;
      // TODO: Store bbox ID in particle state
      return true;
    }

    // Ray intersection test
    float t_near, t_far;
    if (bbox.ray_intersects(particle.position, particle.direction, t_near,
                            t_far)) {
      if (t_near < 10.0f) { // 10 cm look-ahead
        particle.current_level = COARSE_LEVEL_2_SIMPLIFIED_GEO;
        // TODO: Store bbox ID in particle state
        return true;
      }
    }
  }

  // Particle failed Level 1 test - no relevant bounding boxes found
  particle.active = false;
  particle.exit_reason = EXIT_BOUNDARY;
  return false;
}

// ============================================================================
// LEVEL 2: SIMPLIFIED GEOMETRY APPROXIMATION
// ============================================================================

CUDA_HOST_DEVICE
bool level2_simplified_geometry_test(
    particle_filter_state_t &particle,
    const simplified_geometry_t *simplified_geometries,
    uint32_t num_geometries) {
  if (num_geometries == 0 || !simplified_geometries) {
    particle.current_level = COARSE_LEVEL_3_DETAILED_CALC;
    return true; // No simplified geometries, proceed to detailed calculation
  }

  // Check against simplified geometries
  for (uint32_t i = 0; i < num_geometries; ++i) {
    const simplified_geometry_t &geo = simplified_geometries[i];

    // Quick containment test
    if (geo.contains_point(particle.position)) {
      // Estimate intersection distance
      float approx_distance = geo.approximate_ray_intersection(
          particle.position, particle.direction);

      if (approx_distance > 0.0f && approx_distance < 5.0f) { // 5 cm threshold
        particle.current_level = COARSE_LEVEL_3_DETAILED_CALC;

        // Apply density-based energy attenuation
        particle.energy *=
            (1.0f - 0.01f * geo.density_factor * approx_distance);

        return true;
      }
    }
  }

  // Check for early exit based on simplified geometry approximation
  if (particle.should_terminate_early()) {
    particle.active = false;
    particle.exit_reason = particle.get_exit_reason();
    return false;
  }

  // Passed simplified geometry test
  particle.current_level = COARSE_LEVEL_3_DETAILED_CALC;
  return true;
}

// ============================================================================
// LEVEL 3: DETAILED CALCULATION PIPELINE
// ============================================================================

template <typename R>
CUDA_HOST_DEVICE bool
level3_detailed_calculation(particle_filter_state_t &particle,
                            track_t<R> &track) {
  // Final early exit check before detailed calculation
  if (evaluate_early_exit(particle)) {
    return false;
  }

  // Update track state with filtered particle information
  track.vtx1.pos.x = static_cast<R>(particle.position.x);
  track.vtx1.pos.y = static_cast<R>(particle.position.y);
  track.vtx1.pos.z = static_cast<R>(particle.position.z);
  track.vtx1.dir.x = static_cast<R>(particle.direction.x);
  track.vtx1.dir.y = static_cast<R>(particle.direction.y);
  track.vtx1.dir.z = static_cast<R>(particle.direction.z);
  track.vtx1.ke = static_cast<R>(particle.energy);

  // Increment depth counter
  particle.depth++;

  // Check depth limit
  if (particle.depth > MAX_DEPTH_THRESHOLD) {
    particle.active = false;
    particle.exit_reason = EXIT_DEPTH_LIMIT;
    return false;
  }

  // Ready for detailed physics calculation
  particle.current_level = COARSE_LEVEL_3_DETAILED_CALC;
  return true;
}

// ============================================================================
// MAIN COARSE-TO-FINE PIPELINE
// ============================================================================

template <typename R>
CUDA_DEVICE void coarse_to_fine_filter_pipeline(
    particle_filter_state_t *particles, track_t<R> *tracks,
    uint32_t num_particles, warp_filter_state_t &warp_state,
    const bounding_box_t *bounding_boxes, uint32_t num_bboxes,
    const simplified_geometry_t *simplified_geometries,
    uint32_t num_geometries) {
#ifdef __CUDACC__
  uint32_t lane_id = threadIdx.x % WARP_SIZE;
#else
  uint32_t lane_id = 0; // For CPU version, assume single thread per warp
#endif

  // Initialize warp state
  if (lane_id == 0) {
    warp_state.initialize_warp();
  }
#ifdef __CUDACC__
  __syncthreads();
#else
  // No synchronization needed for CPU version
#endif

// Get current particle
#ifdef __CUDACC__
  particle_filter_state_t &particle = particles[threadIdx.x];
#else
  particle_filter_state_t &particle =
      particles[lane_id]; // Use lane_id for CPU version
#endif
#ifdef __CUDACC__
  track_t<R> &track = tracks[threadIdx.x];
#else
  track_t<R> &track = tracks[lane_id]; // Use lane_id for CPU version
#endif

  // Skip if particle is not active
  if (!particle.active) {
    warp_state.update_thread_stats(lane_id, particle);
    return;
  }

  // Initialize particle from track if needed
  if (particle.current_level == 0) {
    particle.initialize_from_track(track);
  }

  // Level 1: Bounding box test
  if (particle.current_level <= COARSE_LEVEL_1_BOUNDING_BOX) {
    if (!level1_bounding_box_test(particle, bounding_boxes, num_bboxes)) {
      warp_state.update_thread_stats(lane_id, particle);
      return;
    }
  }

  // Level 2: Simplified geometry test
  if (particle.current_level <= COARSE_LEVEL_2_SIMPLIFIED_GEO) {
    if (!level2_simplified_geometry_test(particle, simplified_geometries,
                                         num_geometries)) {
      warp_state.update_thread_stats(lane_id, particle);
      return;
    }
  }

  // Level 3: Detailed calculation preparation
  if (particle.current_level <= COARSE_LEVEL_3_DETAILED_CALC) {
    if (!level3_detailed_calculation(particle, track)) {
      warp_state.update_thread_stats(lane_id, particle);
      return;
    }
  }

  // Update warp statistics
  warp_state.update_thread_stats(lane_id, particle);
}

// ============================================================================
// WARP-LEVEL CONSISTENCY OPTIMIZATION
// ============================================================================

CUDA_DEVICE
void optimize_warp_consistency(particle_filter_state_t *particles,
                               warp_filter_state_t &warp_state,
                               uint32_t lane_id) {
  // Get dominant level for this warp
  inspection_level_t dominant_level = warp_state.get_dominant_level();

  // Check if warp has significant activity
  if (!warp_state.has_significant_activity()) {
    // Consider terminating entire warp if activity is too low
    if (lane_id == 0) {
      warp_state.active_mask = 0x0;
    }
    return;
  }

// Align particles to dominant level where possible
#ifdef __CUDACC__
  particle_filter_state_t &particle = particles[threadIdx.x];
#else
  particle_filter_state_t &particle =
      particles[lane_id]; // Use lane_id for CPU version
#endif

  if (particle.active && particle.current_level != dominant_level) {
    // Consider promoting or demoting particle based on warp behavior
    if (dominant_level == COARSE_LEVEL_1_BOUNDING_BOX &&
        particle.current_level > COARSE_LEVEL_1_BOUNDING_BOX) {
      // Most particles are still at Level 1, maybe this particle advanced too
      // quickly
      if (particle.energy < EARLY_EXIT_ENERGY_THRESHOLD * 2.0f) {
        particle.current_level = COARSE_LEVEL_1_BOUNDING_BOX;
      }
    }
  }

// Synchronize warp-level decisions
#ifdef __CUDACC__
  __syncthreads();
#else
  // No synchronization needed for CPU version
#endif
}

// ============================================================================
// EARLY EXIT EVALUATION
// ============================================================================

CUDA_HOST_DEVICE
bool evaluate_early_exit(particle_filter_state_t &particle) {
  if (particle.should_terminate_early()) {
    particle.active = false;
    particle.exit_reason = particle.get_exit_reason();
    return true;
  }
  return false;
}

// ============================================================================
// GEOMETRY CREATION UTILITIES
// ============================================================================

bounding_box_t create_bounding_box_from_geometry(const geometry &geometry,
                                                 float padding) {
  // Get geometry position and calculate rough bounds
  vec3<float> pos(static_cast<float>(geometry.pos.x),
                  static_cast<float>(geometry.pos.y),
                  static_cast<float>(geometry.pos.z));

  // Default box dimensions - this should be customized based on geometry type
  vec3<float> dimensions(10.0f, 10.0f, 10.0f); // 10 cm default box

  vec3<float> min_point =
      pos - dimensions * 0.5f - vec3<float>(padding, padding, padding);
  vec3<float> max_point =
      pos + dimensions * 0.5f + vec3<float>(padding, padding, padding);

  return bounding_box_t(min_point, max_point,
                        static_cast<uint32_t>(geometry.geotype));
}

simplified_geometry_t
create_simplified_geometry(const geometry &geometry,
                           uint32_t simplification_level) {
  vec3<float> pos(static_cast<float>(geometry.pos.x),
                  static_cast<float>(geometry.pos.y),
                  static_cast<float>(geometry.pos.z));

  // Adjust size based on simplification level (higher = more simplified =
  // larger)
  float size_factor =
      1.0f + (simplification_level - 1.0f) * 0.2f; // 20% increase per level
  vec3<float> dimensions(5.0f * size_factor, 5.0f * size_factor,
                         5.0f * size_factor);

  return simplified_geometry_t(pos, dimensions, 0,
                               1.0f); // Box type, unit density
}

// ============================================================================
// STATISTICS AND MONITORING
// ============================================================================

coarse_to_fine_stats_t get_coarse_to_fine_statistics() {
  return g_performance_stats;
}

void reset_coarse_to_fine_statistics() { g_performance_stats.reset(); }

void print_coarse_to_fine_performance_summary() {
  g_performance_stats.calculate_hit_ratios();

  std::cout << "\n=== Coarse-to-Fine Performance Summary ===" << std::endl;
  std::cout << "Total particles processed: "
            << g_performance_stats.total_particles_processed << std::endl;
  std::cout << "Level 1 passes: " << g_performance_stats.level1_passes << " ("
            << (g_performance_stats.level_hit_ratios[1] * 100.0f) << "%)"
            << std::endl;
  std::cout << "Level 2 passes: " << g_performance_stats.level2_passes << " ("
            << (g_performance_stats.level_hit_ratios[2] * 100.0f) << "%)"
            << std::endl;
  std::cout << "Level 3 processed: " << g_performance_stats.level3_processed
            << " (" << (g_performance_stats.level_hit_ratios[3] * 100.0f)
            << "%)" << std::endl;

  std::cout << "\nEarly exits:" << std::endl;
  std::cout << "  Energy threshold: "
            << g_performance_stats.early_exits[EXIT_ENERGY_THRESHOLD]
            << std::endl;
  std::cout << "  Statistical: "
            << g_performance_stats.early_exits[EXIT_STATISTICAL] << std::endl;
  std::cout << "  Depth limit: "
            << g_performance_stats.early_exits[EXIT_DEPTH_LIMIT] << std::endl;
  std::cout << "  Boundary: " << g_performance_stats.early_exits[EXIT_BOUNDARY]
            << std::endl;

  if (g_performance_stats.total_particles_processed > 0) {
    std::cout << "Average processing time: "
              << g_performance_stats.average_processing_time_us
              << " μs per particle" << std::endl;
  }
  std::cout << "==========================================\n" << std::endl;
}

// Explicit template instantiations
template CUDA_HOST_DEVICE bool
level3_detailed_calculation<float>(particle_filter_state_t &, track_t<float> &);
template CUDA_HOST_DEVICE bool
level3_detailed_calculation<double>(particle_filter_state_t &,
                                    track_t<double> &);

template CUDA_DEVICE void coarse_to_fine_filter_pipeline<float>(
    particle_filter_state_t *, track_t<float> *, uint32_t,
    warp_filter_state_t &, const bounding_box_t *, uint32_t,
    const simplified_geometry_t *, uint32_t);
template CUDA_DEVICE void coarse_to_fine_filter_pipeline<double>(
    particle_filter_state_t *, track_t<double> *, uint32_t,
    warp_filter_state_t &, const bounding_box_t *, uint32_t,
    const simplified_geometry_t *, uint32_t);

} // namespace mqi
