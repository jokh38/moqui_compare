#ifndef MQI_PHASE4_GEOMETRY_INTERFACE_HPP
#define MQI_PHASE4_GEOMETRY_INTERFACE_HPP

/// \file
///
/// Phase 4: Geometry Integration - Abstract Geometry Interface
///
/// This header defines the abstract geometry interface for the Voxel-BVH hybrid
/// system, providing runtime-selectable geometry modules with
/// performance/accuracy trade-offs.

#include "mqi_common.hpp"
#include "mqi_geometry.hpp"
#include "mqi_grid3d.hpp"
#include "mqi_material.hpp"
#include "mqi_vec.hpp"
#include <array>
#include <memory>
#include <vector>

namespace mqi {

/// Geometry precision levels for performance/accuracy trade-offs
typedef enum {
  FAST_APPROXIMATION = 0, ///< Fast approximate intersection, lower accuracy
  BALANCED = 1,           ///< Balanced performance and accuracy
  HIGH_PRECISION = 2      ///< High precision intersection, slower
} geometry_precision_t;

/// Geometry complexity levels for dynamic LOD
typedef enum {
  COARSE = 0, ///< Coarse representation (voxel mask only)
  MEDIUM = 1, ///< Medium detail (simplified BVH)
  FINE = 2    ///< Fine detail (full BVH)
} geometry_complexity_t;

/// Enhanced intersection result with normal and material information
template <typename R> struct intersect_result_t {
  R dist;                              ///< Distance to intersection
  vec3<R> normal;                      ///< Surface normal at intersection
  vec3<ijk_t> cell;                    ///< Grid cell index
  cell_side side;                      ///< Which side was hit
  transport_type type;                 ///< Type of geometry
  uint32_t material_id;                ///< Material ID at intersection
  bool hit;                            ///< Whether intersection occurred
  geometry_precision_t precision_used; ///< Precision level used

  CUDA_HOST_DEVICE
  intersect_result_t()
      : dist(-1.0), normal(0, 0, 1), cell(-1, -1, -1), side(NONE_XYZ_PLANE),
        type(NORMAL_PHYSICS), material_id(0), hit(false),
        precision_used(BALANCED) {}
};

/// Abstract geometry interface for Phase 4 integration
template <typename R> class geometry_interface {
public:
  const geometry_type geotype;      ///< Geometry type identifier
  geometry_precision_t precision;   ///< Current precision level
  geometry_complexity_t complexity; ///< Current complexity level

  /// Constructor
  CUDA_HOST_DEVICE
  geometry_interface(geometry_type type, geometry_precision_t prec = BALANCED,
                     geometry_complexity_t comp = MEDIUM)
      : geotype(type), precision(prec), complexity(comp) {}

  /// Virtual destructor
  virtual CUDA_HOST_DEVICE ~geometry_interface() = default;

  /// Core geometry operations

  /// Find intersection between ray and geometry
  /// \param origin Ray origin point
  /// \param direction Ray direction (should be normalized)
  /// \param max_distance Maximum search distance
  /// \param precision Desired precision for this query
  /// \return Intersection result with distance, normal, and material info
  virtual CUDA_HOST_DEVICE intersect_result_t<R>
  intersect(const vec3<R> &origin, const vec3<R> &direction,
            R max_distance = std::numeric_limits<R>::max(),
            geometry_precision_t precision = BALANCED) const = 0;

  /// Calculate surface normal at given point
  /// \param point Point on surface
  /// \param epsilon Small offset for numerical stability
  /// \return Surface normal vector (normalized)
  virtual CUDA_HOST_DEVICE vec3<R>
  calculate_normal(const vec3<R> &point,
                   R epsilon = static_cast<R>(1e-6)) const = 0;

  /// Query material properties at given point
  /// \param point Point in space
  /// \return Material ID
  virtual CUDA_HOST_DEVICE uint32_t
  get_material_id(const vec3<R> &point) const = 0;

  /// Query material density at given point
  /// \param point Point in space
  /// \return Material density (g/cm³)
  virtual CUDA_HOST_DEVICE R
  get_material_density(const vec3<R> &point) const = 0;

  /// Coarse filtering for early rejection (Level 1)
  /// \param aabb Axis-aligned bounding box to test
  /// \return True if geometry might intersect with AABB
  virtual CUDA_HOST_DEVICE bool
  coarse_intersect_aabb(const vec3<R> &aabb_min,
                        const vec3<R> &aabb_max) const = 0;

  /// Simplified geometry test (Level 2)
  /// \param origin Ray origin
  /// \param direction Ray direction
  /// \return True if ray might intersect detailed geometry
  virtual CUDA_HOST_DEVICE bool
  simplified_intersect(const vec3<R> &origin,
                       const vec3<R> &direction) const = 0;

  /// Performance and configuration

  /// Set precision level for performance/accuracy trade-off
  /// \param new_precision New precision level
  virtual CUDA_HOST_DEVICE void
  set_precision(geometry_precision_t new_precision) {
    precision = new_precision;
  }

  /// Set complexity level for dynamic LOD
  /// \param new_complexity New complexity level
  virtual CUDA_HOST_DEVICE void
  set_complexity(geometry_complexity_t new_complexity) {
    complexity = new_complexity;
  }

  /// Get estimated performance cost
  /// \return Relative performance cost (1.0 = baseline)
  virtual CUDA_HOST_DEVICE float get_performance_cost() const = 0;

  /// Get estimated accuracy
  /// \return Relative accuracy (1.0 = perfect)
  virtual CUDA_HOST_DEVICE float get_accuracy_estimate() const = 0;

  /// Memory management

  /// Estimate memory usage in bytes
  /// \return Estimated memory usage
  virtual CUDA_HOST_DEVICE size_t estimate_memory_usage() const = 0;

  /// Prepare geometry for GPU rendering
  /// \return True if preparation successful
  virtual bool prepare_for_gpu() = 0;

  /// Cleanup GPU resources
  virtual void cleanup_gpu() = 0;

  /// Utility functions

  /// Transform point to local coordinates
  /// \param world_point Point in world coordinates
  /// \return Point in local geometry coordinates
  virtual CUDA_HOST_DEVICE vec3<R>
  world_to_local(const vec3<R> &world_point) const = 0;

  /// Transform point to world coordinates
  /// \param local_point Point in local coordinates
  /// \return Point in world coordinates
  virtual CUDA_HOST_DEVICE vec3<R>
  local_to_world(const vec3<R> &local_point) const = 0;

  /// Check if point is inside geometry volume
  /// \param point Point to test
  /// \return True if point is inside
  virtual CUDA_HOST_DEVICE bool is_inside(const vec3<R> &point) const = 0;

  /// Get bounding box in world coordinates
  /// \param aabb_min Output for minimum corner
  /// \param aabb_max Output for maximum corner
  virtual void get_world_bounds(vec3<R> &aabb_min, vec3<R> &aabb_max) const = 0;

  /// Debug and visualization

  /// Print geometry information (debug only)
  virtual void dump_info() const = 0;

  /// Validate geometry integrity
  /// \return True if geometry is valid
  virtual bool validate() const = 0;
};

/// Geometry factory for creating different geometry types
template <typename R> class geometry_factory {
public:
  /// Create geometry instance by type
  /// \param type Geometry type to create
  /// \param precision Initial precision level
  /// \param complexity Initial complexity level
  /// \return Smart pointer to geometry instance
  static std::unique_ptr<geometry_interface<R>>
  create_geometry(geometry_type type, geometry_precision_t precision = BALANCED,
                  geometry_complexity_t complexity = MEDIUM);

  /// Register custom geometry type
  /// \param type Geometry type identifier
  /// \param creator Function to create geometry instance
  static void register_geometry_type(
      geometry_type type,
      std::unique_ptr<geometry_interface<R>> (*creator)(geometry_precision_t,
                                                        geometry_complexity_t));

  /// Get list of available geometry types
  /// \return Vector of available geometry types
  static std::vector<geometry_type> get_available_types();

  /// Check if geometry type is supported
  /// \param type Geometry type to check
  /// \return True if supported
  static bool is_type_supported(geometry_type type);
};

/// Geometry manager for handling multiple geometry instances
template <typename R> class geometry_manager {
private:
  std::vector<std::unique_ptr<geometry_interface<R>>> geometries_;
  geometry_precision_t global_precision_;
  geometry_complexity_t global_complexity_;

public:
  /// Constructor
  geometry_manager(geometry_precision_t precision = BALANCED,
                   geometry_complexity_t complexity = MEDIUM)
      : global_precision_(precision), global_complexity_(complexity) {}

  /// Add geometry to manager
  /// \param geometry Geometry instance to add (takes ownership)
  void add_geometry(std::unique_ptr<geometry_interface<R>> geometry);

  /// Remove geometry by index
  /// \param index Index of geometry to remove
  void remove_geometry(size_t index);

  /// Get geometry by index
  /// \param index Index of geometry
  /// \return Pointer to geometry instance
  geometry_interface<R> *get_geometry(size_t index);

  /// Get number of geometries
  /// \return Number of managed geometries
  size_t get_geometry_count() const { return geometries_.size(); }

  /// Find nearest intersection among all geometries
  /// \param origin Ray origin
  /// \param direction Ray direction
  /// \param max_distance Maximum search distance
  /// \return Nearest intersection result
  intersect_result_t<R> find_nearest_intersection(
      const vec3<R> &origin, const vec3<R> &direction,
      R max_distance = std::numeric_limits<R>::max()) const;

  /// Set global precision for all geometries
  /// \param precision New precision level
  void set_global_precision(geometry_precision_t precision);

  /// Set global complexity for all geometries
  /// \param complexity New complexity level
  void set_global_complexity(geometry_complexity_t complexity);

  /// Prepare all geometries for GPU
  /// \return True if all prepared successfully
  bool prepare_all_for_gpu();

  /// Cleanup all GPU resources
  void cleanup_all_gpu();

  /// Get total memory usage estimate
  /// \return Total estimated memory usage in bytes
  size_t get_total_memory_usage() const;

  /// Validate all geometries
  /// \return True if all geometries are valid
  bool validate_all() const;
};

} // namespace mqi

#endif // MQI_PHASE4_GEOMETRY_INTERFACE_HPP
