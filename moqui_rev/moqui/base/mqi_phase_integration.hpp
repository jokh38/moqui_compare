#ifndef MQI_PHASE3_INTEGRATION_HPP
#define MQI_PHASE3_INTEGRATION_HPP

/// \file
/// \brief Phase 3.0 Integration Layer - Connecting Coarse-to-Fine Framework
/// with Existing Phases
///
/// This header provides the integration layer that connects the Phase 3
/// coarse-to-fine acceleration framework with the existing Phase 1 (persistent
/// threads) and Phase 2 (memory optimization) infrastructure. It demonstrates
/// how the new acceleration framework fits into the overall Moqui Coarse2Fine
/// architecture.

#include "mqi_persistent_threads.hpp"
#include "mqi_integration.hpp"
#include "mqi_memory_optimization.hpp"
#include "mqi_coarse_to_fine.hpp"
#include "mqi_track.hpp"
#include "mqi_work_queue.hpp"
#include <memory>
#include <vector>

namespace mqi {

// ============================================================================
// INTEGRATED PARTICLE TRANSPORT ENGINE
// ============================================================================

/// Integrated particle transport engine combining all three phases
class integrated_transport_engine_t {
private:
  // Phase 1 components
  std::unique_ptr<persistent_threads_t> thread_pool_;
  std::unique_ptr<work_queue_t> work_queue_;

  // Phase 2 components
  segmented_memory_pool_t memory_pool_;
  memory_performance_t memory_performance_;

  // Phase 3 components
  std::vector<bounding_box_t> bounding_boxes_;
  std::vector<simplified_geometry_t> simplified_geometries_;
  coarse_to_fine_stats_t performance_stats_;

  // Engine state
  bool initialized_;
  uint32_t num_particles_;
  uint32_t device_id_;

public:
  /// Constructor
  integrated_transport_engine_t()
      : initialized_(false), num_particles_(0), device_id_(0) {
    ;
  }

  /// Destructor
  ~integrated_transport_engine_t() { shutdown(); }

  /// Initialize the integrated transport engine
  /**
   * \param num_particles Number of particles to process in parallel
   * \param device_id GPU device ID (0 for default)
   * \param physics_pool_size_mb Physics tables pool size in MB
   * \param particle_pool_size_mb Particle data pool size in MB
   * \return true if initialization successful, false otherwise
   */
  bool initialize(uint32_t num_particles = 10000, uint32_t device_id = 0,
                  uint32_t physics_pool_size_mb = 32,
                  uint32_t particle_pool_size_mb = 64) {

    if (initialized_) {
      return true; // Already initialized
    }

    num_particles_ = num_particles;
    device_id_ = device_id;

    std::cout << "Initializing Integrated Transport Engine..." << std::endl;

    // Initialize Phase 1: Persistent threads
    thread_pool_ = std::make_unique<persistent_threads_t>();
    if (!thread_pool_->initialize(num_particles)) {
      std::cerr << "Failed to initialize persistent threads" << std::endl;
      return false;
    }

    work_queue_ = std::make_unique<work_queue_t>();
    if (!work_queue_->initialize(num_particles * 2)) { // 2x capacity
      std::cerr << "Failed to initialize work queue" << std::endl;
      return false;
    }

    std::cout << "  Phase 1: Persistent threads initialized" << std::endl;

    // Initialize Phase 2: Memory optimization
    if (!initialize_phase2_memory_optimization(
            &memory_pool_, physics_pool_size_mb, particle_pool_size_mb,
            true, // Use unified memory
            device_id)) {
      std::cerr << "Failed to initialize Phase 2 memory optimization"
                << std::endl;
      return false;
    }

    std::cout << "  Phase 2: Memory optimization initialized" << std::endl;
    std::cout << "    Physics pool: " << physics_pool_size_mb << " MB"
              << std::endl;
    std::cout << "    Particle pool: " << particle_pool_size_mb << " MB"
              << std::endl;

    // Initialize Phase 3: Coarse-to-fine framework
    if (!initialize_coarse_to_fine_framework(
            bounding_boxes_.data(),
            static_cast<uint32_t>(bounding_boxes_.size()),
            simplified_geometries_.data(),
            static_cast<uint32_t>(simplified_geometries_.size()))) {
      std::cerr << "Failed to initialize Phase 3 coarse-to-fine framework"
                << std::endl;
      return false;
    }

    std::cout << "  Phase 3: Coarse-to-fine framework initialized" << std::endl;
    std::cout << "    Bounding boxes: " << bounding_boxes_.size() << std::endl;
    std::cout << "    Simplified geometries: " << simplified_geometries_.size()
              << std::endl;

    // Reset statistics
    reset_coarse_to_fine_statistics();
    reset_memory_performance_statistics(&memory_pool_);

    initialized_ = true;
    std::cout << "Integrated Transport Engine initialization complete!"
              << std::endl;
    return true;
  }

  /// Shutdown the integrated transport engine
  void shutdown() {
    if (!initialized_) {
      return;
    }

    std::cout << "Shutting down Integrated Transport Engine..." << std::endl;

    // Shutdown Phase 3
    shutdown_coarse_to_fine_framework();

    // Shutdown Phase 2
    shutdown_phase2_memory_optimization(&memory_pool_);

    // Shutdown Phase 1
    if (thread_pool_) {
      thread_pool_->shutdown();
    }
    if (work_queue_) {
      work_queue_->shutdown();
    }

    initialized_ = false;
    std::cout << "Integrated Transport Engine shutdown complete" << std::endl;
  }

  /// Add a bounding box to the acceleration structure
  /**
   * \param min_point Minimum corner of the bounding box
   * \param max_point Maximum corner of the bounding box
   * \param geometry_id Associated geometry identifier
   * \param material_id Material identifier
   * \param importance_weight Statistical importance weight
   */
  void add_bounding_box(const vec3<float> &min_point,
                        const vec3<float> &max_point, uint32_t geometry_id = 0,
                        uint32_t material_id = 0,
                        float importance_weight = 1.0f) {

    bounding_boxes_.emplace_back(min_point, max_point, geometry_id, material_id,
                                 importance_weight);

    // Reinitialize Phase 3 if already initialized
    if (initialized_) {
      shutdown_coarse_to_fine_framework();
      initialize_coarse_to_fine_framework(
          bounding_boxes_.data(), static_cast<uint32_t>(bounding_boxes_.size()),
          simplified_geometries_.data(),
          static_cast<uint32_t>(simplified_geometries_.size()));
    }
  }

  /// Add a simplified geometry to the acceleration structure
  /**
   * \param center Center position of the geometry
   * \param dimensions Dimensions (for box-like geometry)
   * \param parent_bbox Parent bounding box ID
   * \param density_factor Relative density factor
   */
  void add_simplified_geometry(const vec3<float> &center,
                               const vec3<float> &dimensions,
                               uint32_t parent_bbox = 0,
                               float density_factor = 1.0f) {

    simplified_geometries_.emplace_back(center, dimensions, parent_bbox,
                                        density_factor);

    // Reinitialize Phase 3 if already initialized
    if (initialized_) {
      shutdown_coarse_to_fine_framework();
      initialize_coarse_to_fine_framework(
          bounding_boxes_.data(), static_cast<uint32_t>(bounding_boxes_.size()),
          simplified_geometries_.data(),
          static_cast<uint32_t>(simplified_geometries_.size()));
    }
  }

  /// Add a spherical simplified geometry
  /**
   * \param center Center position of the sphere
   * \param radius Sphere radius
   * \param parent_bbox Parent bounding box ID
   * \param density_factor Relative density factor
   */
  void add_simplified_sphere(const vec3<float> &center, float radius,
                             uint32_t parent_bbox = 0,
                             float density_factor = 1.0f) {

    simplified_geometries_.emplace_back(center, radius, parent_bbox,
                                        density_factor);

    // Reinitialize Phase 3 if already initialized
    if (initialized_) {
      shutdown_coarse_to_fine_framework();
      initialize_coarse_to_fine_framework(
          bounding_boxes_.data(), static_cast<uint32_t>(bounding_boxes_.size()),
          simplified_geometries_.data(),
          static_cast<uint32_t>(simplified_geometries_.size()));
    }
  }

  /// Process a batch of particles through the integrated pipeline
  /**
   * \tparam R Floating point type (float or double)
   * \param particles Array of particle tracks to process
   * \param num_particles Number of particles in the batch
   * \return true if processing successful, false otherwise
   */
  template <typename R>
  bool process_particle_batch(track_t<R> *particles, uint32_t num_particles) {
    if (!initialized_) {
      std::cerr << "Engine not initialized" << std::endl;
      return false;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Allocate particle filter states from Phase 2 memory pool
    particle_filter_state_t *filter_states =
        static_cast<particle_filter_state_t *>(allocate_from_particle_pool(
            &memory_pool_, num_particles * sizeof(particle_filter_state_t),
            64));

    if (!filter_states) {
      std::cerr << "Failed to allocate particle filter states" << std::endl;
      return false;
    }

    // Initialize filter states from tracks
    for (uint32_t i = 0; i < num_particles; ++i) {
      filter_states[i].initialize_from_track(particles[i]);
    }

    // Process through Phase 3 coarse-to-fine pipeline
    // This would typically be launched on GPU, but for demonstration we use CPU
    for (uint32_t i = 0; i < num_particles; ++i) {
      particle_filter_state_t &particle = filter_states[i];

      // Level 1: Bounding box test
      if (!level1_bounding_box_test(
              particle, bounding_boxes_.data(),
              static_cast<uint32_t>(bounding_boxes_.size()))) {
        continue; // Particle terminated
      }

      // Level 2: Simplified geometry test
      if (!level2_simplified_geometry_test(
              particle, simplified_geometries_.data(),
              static_cast<uint32_t>(simplified_geometries_.size()))) {
        continue; // Particle terminated
      }

      // Level 3: Detailed calculation
      if (!level3_detailed_calculation(particle, particles[i])) {
        continue; // Particle terminated
      }

      // Here you would call the actual physics calculation kernels
      // This is where the detailed Monte Carlo transport happens
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);

    std::cout << "Processed " << num_particles << " particles in "
              << duration.count() << " μs" << std::endl;
    std::cout << "Average: " << (duration.count() / num_particles)
              << " μs per particle" << std::endl;

    // Update performance statistics
    performance_stats_ = get_coarse_to_fine_statistics();
    performance_stats_.total_particles_processed += num_particles;
    performance_stats_.average_processing_time_us =
        static_cast<float>(duration.count()) / num_particles;

    // Memory performance statistics
    get_memory_performance_statistics(&memory_pool_, &memory_performance_);

    return true;
  }

  /// Get comprehensive performance statistics
  /**
   * \return Structure containing all performance metrics
   */
  struct comprehensive_performance_stats_t {
    coarse_to_fine_stats_t phase3_stats;
    memory_performance_t memory_stats;
    uint32_t total_particles_processed;
    float average_processing_time_us;
    float memory_bandwidth_utilization_gb_s;
  };

  comprehensive_performance_stats_t
  get_comprehensive_performance_stats() const {
    comprehensive_performance_stats_t stats;
    stats.phase3_stats = performance_stats_;
    stats.memory_stats = memory_performance_;
    stats.total_particles_processed =
        performance_stats_.total_particles_processed;
    stats.average_processing_time_us =
        performance_stats_.average_processing_time_us;
    stats.memory_bandwidth_utilization_gb_s =
        memory_performance_.bandwidth_utilization_gb_s;
    return stats;
  }

  /// Print comprehensive performance report
  void print_performance_report() const {
    std::cout << "\n=== Comprehensive Performance Report ===" << std::endl;
    std::cout << "Total particles processed: "
              << performance_stats_.total_particles_processed << std::endl;
    std::cout << "Average processing time: "
              << performance_stats_.average_processing_time_us << " μs/particle"
              << std::endl;

    std::cout << "\nPhase 3 Coarse-to-Fine Statistics:" << std::endl;
    performance_stats_.calculate_hit_ratios();
    std::cout << "  Level 1 hit ratio: "
              << (performance_stats_.level_hit_ratios[1] * 100.0f) << "%"
              << std::endl;
    std::cout << "  Level 2 hit ratio: "
              << (performance_stats_.level_hit_ratios[2] * 100.0f) << "%"
              << std::endl;
    std::cout << "  Level 3 hit ratio: "
              << (performance_stats_.level_hit_ratios[3] * 100.0f) << "%"
              << std::endl;

    std::cout << "\nPhase 2 Memory Performance:" << std::endl;
    std::cout << "  Bandwidth utilization: "
              << memory_performance_.bandwidth_utilization_gb_s << " GB/s"
              << std::endl;
    std::cout << "  Cache hit ratio: "
              << (memory_performance_.cache_hit_ratio * 100.0f) << "%"
              << std::endl;
    std::cout << "  Memory pool utilization: "
              << memory_performance_.memory_pool_utilization << "%"
              << std::endl;

    std::cout << "==========================================\n" << std::endl;
  }

  /// Check if engine is properly initialized
  bool is_initialized() const { return initialized_; }

  /// Get number of particles the engine can handle
  uint32_t get_capacity() const { return num_particles_; }

  /// Get current device ID
  uint32_t get_device_id() const { return device_id_; }
};

// ============================================================================
// UTILITY FUNCTIONS FOR INTEGRATED WORKFLOW
// ============================================================================

/// Create a simple water phantom geometry for testing
/**
 * \param engine Integrated transport engine to configure
 * \param phantom_size Size of the water phantom (cm)
 * \param voxel_size Size of voxels (cm)
 */
void setup_water_phantom_test(integrated_transport_engine_t &engine,
                              float phantom_size = 30.0f,
                              float voxel_size = 1.0f) {

  // Create bounding box for water phantom
  vec3<float> phantom_min(-phantom_size * 0.5f, -phantom_size * 0.5f,
                          -phantom_size * 0.5f);
  vec3<float> phantom_max(phantom_size * 0.5f, phantom_size * 0.5f,
                          phantom_size * 0.5f);
  engine.add_bounding_box(phantom_min, phantom_max, 1, 1,
                          1.0f); // Water material

  // Create simplified geometry representing the phantom
  vec3<float> phantom_center(0.0f, 0.0f, 0.0f);
  vec3<float> phantom_dims(phantom_size, phantom_size, phantom_size);
  engine.add_simplified_geometry(phantom_center, phantom_dims, 0, 1.0f);

  std::cout << "Water phantom test geometry configured:" << std::endl;
  std::cout << "  Size: " << phantom_size << " cm" << std::endl;
  std::cout << "  Voxel size: " << voxel_size << " cm" << std::endl;
}

/// Create a multi-region test geometry
/**
 * \param engine Integrated transport engine to configure
 * \param regions Vector of region parameters (center, size, density)
 */
void setup_multi_region_test(
    integrated_transport_engine_t &engine,
    const std::vector<std::tuple<vec3<float>, vec3<float>, float>> &regions) {

  for (size_t i = 0; i < regions.size(); ++i) {
    const auto &[center, size, density] = regions[i];

    // Add bounding box
    vec3<float> min_point = center - size * 0.5f;
    vec3<float> max_point = center + size * 0.5f;
    engine.add_bounding_box(min_point, max_point, static_cast<uint32_t>(i),
                            static_cast<uint32_t>(i), density);

    // Add simplified geometry
    engine.add_simplified_geometry(center, size, static_cast<uint32_t>(i),
                                   density);
  }

  std::cout << "Multi-region test geometry configured with " << regions.size()
            << " regions" << std::endl;
}

} // namespace mqi

#endif // MQI_PHASE3_INTEGRATION_HPP
