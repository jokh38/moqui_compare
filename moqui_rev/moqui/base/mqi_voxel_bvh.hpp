#ifndef MQI_PHASE4_VOXEL_BVH_HPP
#define MQI_PHASE4_VOXEL_BVH_HPP

/// \file
///
/// Phase 4: Voxel-BVH Hybrid System
///
/// This header implements a hybrid voxel mask and BVH (Bounding Volume
/// Hierarchy) system for efficient geometry intersection testing with
/// coarse-to-fine filtering.

#include "mqi_common.hpp"
#include "mqi_geometry_interface.hpp"
#include "mqi_vec.hpp"
#include <array>
#include <memory>
#include <vector>

namespace mqi {

// Forward declarations
template <typename R> struct bvh_node;
template <typename R> struct voxel_mask;

/// BVH node structure for GPU-friendly layout
template <typename R> struct alignas(16) bvh_node {
  vec3<R> bounds_min;       ///< Minimum bounds of this node
  vec3<R> bounds_max;       ///< Maximum bounds of this node
  uint32_t left_child;      ///< Index of left child (0 for leaf)
  uint32_t right_child;     ///< Index of right child (0 for leaf)
  uint32_t primitive_start; ///< Start index for primitives (leaf only)
  uint32_t primitive_count; ///< Number of primitives (leaf only)
  uint32_t geometry_id;     ///< Geometry identifier (leaf only)
  uint8_t is_leaf;          ///< 1 if leaf node, 0 if internal
  uint8_t pad[7];           ///< Padding for 16-byte alignment

  CUDA_HOST_DEVICE
  bvh_node()
      : bounds_min(0), bounds_max(0), left_child(0), right_child(0),
        primitive_start(0), primitive_count(0), geometry_id(0), is_leaf(0) {
    pad[0] = pad[1] = pad[2] = pad[3] = pad[4] = pad[5] = pad[6] = 0;
  }

  /// Check if ray intersects this node's bounding box
  CUDA_HOST_DEVICE
  bool intersect_ray(const vec3<R> &origin, const vec3<R> &direction,
                     R t_min = 0,
                     R t_max = std::numeric_limits<R>::max()) const {
    vec3<R> inv_dir(1.0 / direction.x, 1.0 / direction.y, 1.0 / direction.z);

    R tx1 = (bounds_min.x - origin.x) * inv_dir.x;
    R tx2 = (bounds_max.x - origin.x) * inv_dir.x;

    R tmin = std::min(tx1, tx2);
    R tmax = std::max(tx1, tx2);

    R ty1 = (bounds_min.y - origin.y) * inv_dir.y;
    R ty2 = (bounds_max.y - origin.y) * inv_dir.y;

    tmin = std::max(tmin, std::min(ty1, ty2));
    tmax = std::min(tmax, std::max(ty1, ty2));

    R tz1 = (bounds_min.z - origin.z) * inv_dir.z;
    R tz2 = (bounds_max.z - origin.z) * inv_dir.z;

    tmin = std::max(tmin, std::min(tz1, tz2));
    tmax = std::min(tmax, std::max(tz1, tz2));

    return tmax >= tmin && tmin < t_max && tmax > t_min;
  }
};

/// 2D voxel mask for coarse filtering (beam-eye view projection)
template <typename R> struct voxel_mask {
  uint16_t width;                 ///< Mask width in voxels
  uint16_t height;                ///< Mask height in voxels
  R world_width;                  ///< World space width covered
  R world_height;                 ///< World space height covered
  vec3<R> origin;                 ///< World space origin of mask
  std::vector<uint8_t> mask_data; ///< Bit mask data (1 byte per voxel)

  voxel_mask()
      : width(0), height(0), world_width(0), world_height(0), origin(0) {}

  /// Create mask from geometry projection
  void create_from_geometry(const geometry_interface<R> *geometry,
                            const vec3<R> &beam_direction, R distance,
                            uint16_t resolution);

  /// Check if ray passes through mask
  CUDA_HOST_DEVICE
  bool ray_passes_mask(const vec3<R> &origin, const vec3<R> &direction) const {
    if (width == 0 || height == 0)
      return true; // No mask means always pass

    // Project ray onto mask plane
    R t = (origin.z - this->origin.z) / direction.z;
    if (t < 0)
      return false;

    vec3<R> intersection = origin + direction * t;

    // Convert to mask coordinates
    R local_x = intersection.x - this->origin.x;
    R local_y = intersection.y - this->origin.y;

    if (local_x < 0 || local_x >= world_width || local_y < 0 ||
        local_y >= world_height) {
      return false;
    }

    uint16_t mask_x = static_cast<uint16_t>((local_x / world_width) * width);
    uint16_t mask_y = static_cast<uint16_t>((local_y / world_height) * height);

    size_t index = mask_y * width + mask_x;
    return index < mask_data.size() && mask_data[index] != 0;
  }
};

/// Stackless BVH traversal state for GPU efficiency
template <typename R> struct traversal_state {
  uint32_t node_index; ///< Current node index
  R t_min;             ///< Minimum t for this node
  R t_max;             ///< Maximum t for this node

  CUDA_HOST_DEVICE
  traversal_state()
      : node_index(0), t_min(0), t_max(std::numeric_limits<R>::max()) {}

  CUDA_HOST_DEVICE
  traversal_state(uint32_t idx, R tmin, R tmax)
      : node_index(idx), t_min(tmin), t_max(tmax) {}
};

/// Voxel-BVH hybrid geometry implementation
template <typename R> class voxel_bvh_hybrid : public geometry_interface<R> {
private:
  std::vector<bvh_node<R>> bvh_nodes_;
  voxel_mask<R> voxel_mask_;
  geometry_precision_t current_precision_;
  geometry_complexity_t current_complexity_;
  mutable std::vector<traversal_state<R>> traversal_stack_;
  bool gpu_ready_;
  size_t gpu_memory_size_;

public:
  /// Constructor
  voxel_bvh_hybrid(geometry_type type = PATIENT,
                   geometry_precision_t precision = BALANCED,
                   geometry_complexity_t complexity = MEDIUM);

  /// Destructor
  virtual ~voxel_bvh_hybrid() { cleanup_gpu(); }

  // Geometry interface implementation
  virtual CUDA_HOST_DEVICE intersect_result_t<R>
  intersect(const vec3<R> &origin, const vec3<R> &direction,
            R max_distance = std::numeric_limits<R>::max(),
            geometry_precision_t precision = BALANCED) const override;

  virtual CUDA_HOST_DEVICE vec3<R>
  calculate_normal(const vec3<R> &point, R epsilon = 1e-6) const override;

  virtual CUDA_HOST_DEVICE uint32_t
  get_material_id(const vec3<R> &point) const override;

  virtual CUDA_HOST_DEVICE R
  get_material_density(const vec3<R> &point) const override;

  virtual CUDA_HOST_DEVICE bool
  coarse_intersect_aabb(const vec3<R> &aabb_min,
                        const vec3<R> &aabb_max) const override;

  virtual CUDA_HOST_DEVICE bool
  simplified_intersect(const vec3<R> &origin,
                       const vec3<R> &direction) const override;

  virtual CUDA_HOST_DEVICE float get_performance_cost() const override;

  virtual CUDA_HOST_DEVICE float get_accuracy_estimate() const override;

  virtual CUDA_HOST_DEVICE size_t estimate_memory_usage() const override;

  virtual bool prepare_for_gpu() override;

  virtual void cleanup_gpu() override;

  virtual CUDA_HOST_DEVICE vec3<R>
  world_to_local(const vec3<R> &world_point) const override;

  virtual CUDA_HOST_DEVICE vec3<R>
  local_to_world(const vec3<R> &local_point) const override;

  virtual CUDA_HOST_DEVICE bool is_inside(const vec3<R> &point) const override;

  virtual void get_world_bounds(vec3<R> &aabb_min,
                                vec3<R> &aabb_max) const override;

  virtual void dump_info() const override;

  virtual bool validate() const override;

  // BVH-specific methods
  void build_bvh(const std::vector<vec3<R>> &vertices,
                 const std::vector<uint32_t> &indices,
                 const std::vector<uint32_t> &material_ids);

  void update_voxel_mask(const vec3<R> &beam_direction, R distance);

  void set_precision(geometry_precision_t precision) override;

  void set_complexity(geometry_complexity_t complexity) override;

  /// Get BVH statistics
  struct bvh_stats {
    uint32_t total_nodes;
    uint32_t leaf_nodes;
    uint32_t max_depth;
    float average_primitives_per_leaf;
  };

  bvh_stats get_bvh_stats() const;

private:
  // BVH construction methods
  void build_bvh_recursive(uint32_t node_id, uint32_t start, uint32_t end,
                           std::vector<vec3<R>> &centroids,
                           std::vector<uint32_t> &primitive_indices);

  void calculate_node_bounds(uint32_t node_id, uint32_t start, uint32_t end,
                             const std::vector<vec3<R>> &vertices,
                             const std::vector<uint32_t> &indices);

  uint32_t find_best_split(uint32_t start, uint32_t end, int axis,
                           const std::vector<vec3<R>> &centroids) const;

  // Stackless traversal implementation
  CUDA_HOST_DEVICE
  intersect_result_t<R>
  traverse_bvh_stackless(const vec3<R> &origin, const vec3<R> &direction,
                         R max_distance, geometry_precision_t precision) const;

  CUDA_HOST_DEVICE
  intersect_result_t<R> intersect_leaf_primitive(uint32_t primitive_id,
                                                 const vec3<R> &origin,
                                                 const vec3<R> &direction,
                                                 R max_distance) const;

  // GPU memory management
  bool upload_bvh_to_gpu();
  void download_bvh_from_gpu();
  void *gpu_bvh_buffer_;
  void *gpu_mask_buffer_;

  // Coordinate transformation
  vec3<R> transform_origin_;
  vec3<R> transform_scale_;
  mat3x3<R> transform_rotation_;

  // Primitive data
  std::vector<vec3<R>> vertices_;
  std::vector<uint32_t> indices_;
  std::vector<uint32_t> material_ids_;
  std::vector<R> material_densities_;
};

/// Utility functions for BVH operations
namespace voxel_bvh_utils {
/// Calculate optimal split using surface area heuristic (SAH)
template <typename R>
uint32_t calculate_sah_split(const std::vector<bvh_node<R>> &nodes,
                             uint32_t start, uint32_t end, int axis,
                             const std::vector<vec3<R>> &centroids);

/// Calculate bounds for a set of primitives
template <typename R>
void calculate_primitive_bounds(const std::vector<vec3<R>> &vertices,
                                const std::vector<uint32_t> &indices,
                                uint32_t start, uint32_t end,
                                vec3<R> &min_bounds, vec3<R> &max_bounds);

/// Optimize BVH tree structure
template <typename R> void optimize_bvh_tree(std::vector<bvh_node<R>> &nodes);

/// Validate BVH structure
template <typename R> bool validate_bvh(const std::vector<bvh_node<R>> &nodes);

} // namespace voxel_bvh_utils

} // namespace mqi

#endif // MQI_PHASE4_VOXEL_BVH_HPP
