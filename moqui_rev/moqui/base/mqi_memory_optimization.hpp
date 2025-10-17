#ifndef MQI_PHASE2_MEMORY_OPTIMIZATION_HPP
#define MQI_PHASE2_MEMORY_OPTIMIZATION_HPP

/// \file
/// \brief Phase 2.0 Memory Access Optimization for GPU Acceleration
///
/// This header implements advanced memory optimization strategies for the Moqui
/// Coarse2Fine GPU optimization project, focusing on:
/// - Constant memory optimization for frequently accessed physics data
/// - Unified memory management for efficient data transfer
/// - Memory access pattern optimization for improved cache utilization
/// - GPU-aware memory pooling and allocation strategies

#include "mqi_common.hpp"
#include "mqi_material.hpp"
#include "mqi_physics_constants.hpp"
#include <atomic>
#include <cstdint>

namespace mqi {

// ============================================================================
// FORWARD DECLARATIONS
// ============================================================================

// Forward declarations for structures used in function signatures
struct material_properties_device_t;
struct physics_constants_device_t;
struct energy_grid_device_t;

// ============================================================================
// CONSTANTS AND CONFIGURATION
// ============================================================================

/// Maximum number of materials that can be stored in constant memory
constexpr uint32_t MAX_CONSTANT_MATERIALS = 16;

/// Maximum size of energy grid in constant memory (in floats)
constexpr uint32_t MAX_CONSTANT_ENERGY_GRID = 1024;

/// Size of physics constants structure in bytes
constexpr uint32_t PHYSICS_CONSTANTS_SIZE = 512;

/// Memory pool segmentation sizes (in MB)
constexpr uint32_t PHYSICS_TABLES_POOL_SIZE = 32; // Read-only, cache-optimized
constexpr uint32_t PARTICLE_DATA_POOL_SIZE = 64;  // Read-write, streaming
constexpr uint32_t STATISTICS_POOL_SIZE = 8;      // Frequently updated

// ============================================================================
// CONSTANT MEMORY STRUCTURES
// ============================================================================

#ifdef __CUDACC__

/// Material properties optimized for constant memory access
struct __align__(16) material_properties_device_t {
  float mass_density;           ///< g/cm^3
  float radiation_length;       ///< cm
  float nuclear_length;         ///< cm
  float stopping_power_ratio;   ///< Relative to water
  float mean_excitation_energy; ///< eV
  float atomic_number;          ///< Z
  float mass_number;            ///< A
  float density_correction;     ///< Density effect correction
  uint32_t material_id;         ///< Material identifier
  uint32_t padding[3];          ///< Alignment padding
};

/// Physics constants optimized for constant memory
struct __align__(64) physics_constants_device_t {
  float electron_mass;             ///< Electron rest mass (MeV/c^2)
  float proton_mass;               ///< Proton rest mass (MeV/c^2)
  float fine_structure_constant;   ///< α
  float classical_electron_radius; ///< cm
  float avogadro_number;           ///< 1/mol
  float speed_of_light;            ///< cm/s
  float electron_charge;           ///< esu
  float hartree_energy;            ///< eV
  float bohr_radius;               ///< cm
  float rydberg_energy;            ///< eV
  float boltzmann_constant;        ///< eV/K
  float planck_constant;           ///< eV*s
  float pi;                        ///< π
  float sqrt_2;                    ///< √2
  float sqrt_3;                    ///< √3
  float padding[45];               ///< Alignment to 64 bytes
};

/// Energy grid data for fast interpolation
struct __align__(16) energy_grid_device_t {
  float energy_values[MAX_CONSTANT_ENERGY_GRID]; ///< Energy points (MeV)
  float step_sizes[MAX_CONSTANT_ENERGY_GRID]; ///< Step sizes for interpolation
  uint32_t grid_size;                         ///< Number of grid points
  float min_energy;                           ///< Minimum energy (MeV)
  float max_energy;                           ///< Maximum energy (MeV)
  uint32_t padding[2];                        ///< Alignment padding
};

// ============================================================================
// DEVICE CONSTANT MEMORY DECLARATIONS
// ============================================================================

/// Constant memory for common materials
extern __constant__ material_properties_device_t
    g_constant_materials[MAX_CONSTANT_MATERIALS];

/// Constant memory for physics constants
extern __constant__ physics_constants_device_t g_physics_constants;

/// Constant memory for energy grid
extern __constant__ energy_grid_device_t g_energy_grid;

#endif // __CUDACC__

// ============================================================================
// HOST MEMORY MANAGEMENT STRUCTURES
// ============================================================================

/// Segmented memory pool for different access patterns
struct segmented_memory_pool_t {
  void *physics_tables_pool;  ///< Read-only physics data
  void *particle_data_pool;   ///< Particle state data
  void *statistics_pool;      ///< Statistics and counters
  size_t physics_tables_size; ///< Size of physics tables pool
  size_t particle_data_size;  ///< Size of particle data pool
  size_t statistics_size;     ///< Size of statistics pool
  bool use_unified_memory;    ///< Unified memory flag
  uint32_t device_id;         ///< GPU device ID
};

/// Memory performance monitoring
struct memory_performance_t {
  std::atomic<uint64_t> bytes_read;    ///< Total bytes read from device memory
  std::atomic<uint64_t> bytes_written; ///< Total bytes written to device memory
  std::atomic<uint64_t> cache_hits;    ///< Cache hit count
  std::atomic<uint64_t> cache_misses;  ///< Cache miss count
  std::atomic<float> cache_hit_ratio;  ///< Cache hit ratio
  float bandwidth_utilization_gb_s;    ///< Memory bandwidth utilization (GB/s)
  uint32_t memory_pool_utilization;    ///< Pool utilization percentage
  uint32_t padding[7];                 ///< Alignment padding
};

/// Memory access pattern statistics
struct memory_access_stats_t {
  uint32_t sequential_accesses; ///< Sequential memory accesses
  uint32_t random_accesses;     ///< Random memory accesses
  uint32_t strided_accesses;    ///< Strided memory accesses
  float average_stride_length;  ///< Average stride length
  float coalescing_efficiency;  ///< Memory coalescing efficiency
  uint32_t padding[4];          ///< Alignment padding
};

// ============================================================================
// PHASE 2.0 MEMORY OPTIMIZATION API
// ============================================================================

/// Initialize Phase 2.0 memory optimization system
/**
 * \param pool Memory pool structure to initialize
 * \param physics_table_size_mb Size of physics tables pool in MB
 * \param particle_data_size_mb Size of particle data pool in MB
 * \param use_unified_memory Enable unified memory allocation
 * \param device_id GPU device ID (0 for default)
 * \return true if initialization successful, false otherwise
 */
bool initialize_phase2_memory_optimization(
    segmented_memory_pool_t *pool,
    uint32_t physics_table_size_mb = PHYSICS_TABLES_POOL_SIZE,
    uint32_t particle_data_size_mb = PARTICLE_DATA_POOL_SIZE,
    bool use_unified_memory = false, uint32_t device_id = 0);

/// Shutdown Phase 2.0 memory optimization system
/**
 * \param pool Memory pool to cleanup
 */
void shutdown_phase2_memory_optimization(segmented_memory_pool_t *pool);

/// Upload material properties to constant memory
/**
 * \param materials Array of material properties
 * \param num_materials Number of materials to upload
 * \return true if upload successful, false otherwise
 */
bool upload_materials_to_constant_memory(
    const material_properties_device_t *materials, uint32_t num_materials);

/// Upload physics constants to constant memory
/**
 * \param physics_consts Physics constants structure
 * \return true if upload successful, false otherwise
 */
bool upload_physics_constants_to_constant_memory(
    const physics_constants_device_t *physics_consts);

/// Upload energy grid to constant memory
/**
 * \param energy_grid Energy grid data
 * \return true if upload successful, false otherwise
 */
bool upload_energy_grid_to_constant_memory(
    const energy_grid_device_t *energy_grid);

/// Allocate memory from physics tables pool
/**
 * \param pool Memory pool structure
 * \param size Size in bytes to allocate
 * \param alignment Alignment requirement (default 64 bytes)
 * \return Pointer to allocated memory, nullptr if allocation failed
 */
void *allocate_from_physics_pool(segmented_memory_pool_t *pool, size_t size,
                                 size_t alignment = 64);

/// Allocate memory from particle data pool
/**
 * \param pool Memory pool structure
 * \param size Size in bytes to allocate
 * \param alignment Alignment requirement (default 32 bytes)
 * \return Pointer to allocated memory, nullptr if allocation failed
 */
void *allocate_from_particle_pool(segmented_memory_pool_t *pool, size_t size,
                                  size_t alignment = 32);

/// Allocate memory from statistics pool
/**
 * \param pool Memory pool structure
 * \param size Size in bytes to allocate
 * \param alignment Alignment requirement (default 16 bytes)
 * \return Pointer to allocated memory, nullptr if allocation failed
 */
void *allocate_from_statistics_pool(segmented_memory_pool_t *pool, size_t size,
                                    size_t alignment = 16);

/// Prefetch physics data for next work chunk
/**
 * \param work_item Work item to prefetch data for
 * \param cache Pointer to physics cache structure
 * \return true if prefetch successful, false otherwise
 */
bool prefetch_physics_data(const void *work_item, void *cache);

/// Optimize memory access patterns for particle transport
/**
 * \param particles Array of particle data
 * \param num_particles Number of particles
 * \param access_pattern Output access pattern statistics
 */
void optimize_memory_access_patterns(const void *particles,
                                     uint32_t num_particles,
                                     memory_access_stats_t *access_pattern);

/// Get memory performance statistics
/**
 * \param pool Memory pool structure
 * \param performance Output performance statistics
 */
void get_memory_performance_statistics(const segmented_memory_pool_t *pool,
                                       memory_performance_t *performance);

/// Reset memory performance statistics
/**
 * \param pool Memory pool structure
 */
void reset_memory_performance_statistics(segmented_memory_pool_t *pool);

/// Validate memory pool integrity
/**
 * \param pool Memory pool to validate
 * \param validation_output Output string for validation results
 * \param output_size Size of validation output buffer
 * \return true if validation passed, false otherwise
 */
bool validate_memory_pool_integrity(const segmented_memory_pool_t *pool,
                                    char *validation_output,
                                    size_t output_size);

/// Diagnose memory performance issues
/**
 * \param pool Memory pool structure
 * \param diagnostic_output Output string for diagnostic results
 * \param output_size Size of diagnostic output buffer
 */
void diagnose_memory_performance(const segmented_memory_pool_t *pool,
                                 char *diagnostic_output, size_t output_size);

// ============================================================================
// DEVICE-SIDE HELPER FUNCTIONS
// ============================================================================

#ifdef __CUDACC__

/// Get material properties from constant memory (device function)
/**
 * \param material_id Material identifier
 * \return Pointer to material properties in constant memory
 */
CUDA_DEVICE const material_properties_device_t *
get_material_properties_device(uint32_t material_id);

/// Get physics constants from constant memory (device function)
/**
 * \return Pointer to physics constants in constant memory
 */
CUDA_DEVICE const physics_constants_device_t *get_physics_constants_device();

/// Interpolate energy grid value (device function)
/**
 * \param energy Energy value to interpolate
 * \return Interpolated value
 */
CUDA_DEVICE float interpolate_energy_grid(float energy);

/// Optimized material property lookup with caching (device function)
/**
 * \param material_id Material identifier
 * \param property_type Type of property to retrieve
 * \return Property value
 */
CUDA_DEVICE float get_material_property_cached(uint32_t material_id,
                                               uint32_t property_type);

/// Memory coalescing check (device function)
/**
 * \param thread_id Thread ID within warp
 * \param access_pattern Memory access pattern to check
 * \return true if access is coalesced, false otherwise
 */
CUDA_DEVICE bool is_memory_access_coalesced(uint32_t thread_id,
                                            uint32_t access_pattern);

#endif // __CUDACC__

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/// Convert host material structure to device constant memory format
/**
 * \param host_material Host-side material structure
 * \param device_material Output device material structure
 */
template <typename R>
void convert_material_to_device_format(
    const void *host_material, material_properties_device_t *device_material);

/// Create optimized energy grid for constant memory
/**
 * \param min_energy Minimum energy value (MeV)
 * \param max_energy Maximum energy value (MeV)
 * \param num_points Number of grid points
 * \param energy_grid Output energy grid structure
 */
void create_optimized_energy_grid(float min_energy, float max_energy,
                                  uint32_t num_points,
                                  energy_grid_device_t *energy_grid);

/// Calculate optimal memory pool sizes based on simulation parameters
/**
 * \param num_particles Expected number of particles
 * \param num_materials Number of materials in simulation
 * \param geometry_complexity Geometry complexity factor (1-10)
 * \param physics_table_size Output physics table pool size
 * \param particle_data_size Output particle data pool size
 * \param statistics_size Output statistics pool size
 */
void calculate_optimal_pool_sizes(uint32_t num_particles,
                                  uint32_t num_materials,
                                  uint32_t geometry_complexity,
                                  uint32_t &physics_table_size,
                                  uint32_t &particle_data_size,
                                  uint32_t &statistics_size);

} // namespace mqi

#endif // MQI_PHASE2_MEMORY_OPTIMIZATION_HPP
