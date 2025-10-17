#include "mqi_voxel_bvh.hpp"
#include "mqi_math.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stack>

namespace mqi {

// voxel_bvh_hybrid implementation
template <typename R>
voxel_bvh_hybrid<R>::voxel_bvh_hybrid(geometry_type type,
                                      geometry_precision_t precision,
                                      geometry_complexity_t complexity)
    : geometry_interface<R>(type, precision, complexity),
      current_precision_(precision), current_complexity_(complexity),
      gpu_ready_(false), gpu_memory_size_(0), gpu_bvh_buffer_(nullptr),
      gpu_mask_buffer_(nullptr), transform_origin_(0), transform_scale_(1),
      transform_rotation_(mat3x3<R>::identity()) {
  traversal_stack_.reserve(64); // Reserve space for traversal stack
}

template <typename R>
intersect_result_t<R>
voxel_bvh_hybrid<R>::intersect(const vec3<R> &origin, const vec3<R> &direction,
                               R max_distance,
                               geometry_precision_t precision) const {
  intersect_result_t<R> result;

  // Transform to local coordinates
  vec3<R> local_origin = world_to_local(origin);
  vec3<R> local_direction = transform_rotation_ * direction;
  local_direction.normalize();

  // Level 1: Coarse filtering with voxel mask
  if (current_complexity_ >= COARSE &&
      !voxel_mask_.ray_passes_mask(local_origin, local_direction)) {
    result.hit = false;
    return result;
  }

  // Level 2: Simplified geometry test
  if (precision == FAST_APPROXIMATION && current_complexity_ >= MEDIUM) {
    if (!simplified_intersect(local_origin, local_direction)) {
      result.hit = false;
      return result;
    }
  }

  // Level 3: Full BVH traversal
  return traverse_bvh_stackless(local_origin, local_direction, max_distance,
                                precision);
}

template <typename R>
vec3<R> voxel_bvh_hybrid<R>::calculate_normal(const vec3<R> &point,
                                              R epsilon) const {
  vec3<R> local_point = world_to_local(point);
  vec3<R> normal(0, 0, 1);

  // Use central differences to calculate normal
  vec3<R> dx(epsilon, 0, 0);
  vec3<R> dy(0, epsilon, 0);
  vec3<R> dz(0, 0, epsilon);

  R center_dist = is_inside(local_point) ? 0 : -1;
  R dx_dist = is_inside(local_point + dx) ? 0 : -1;
  R dy_dist = is_inside(local_point + dy) ? 0 : -1;
  R dz_dist = is_inside(local_point + dz) ? 0 : -1;

  normal.x = (dx_dist - center_dist) / epsilon;
  normal.y = (dy_dist - center_dist) / epsilon;
  normal.z = (dz_dist - center_dist) / epsilon;
  normal.normalize();

  // Transform normal back to world coordinates
  return transform_rotation_ * normal;
}

template <typename R>
uint32_t voxel_bvh_hybrid<R>::get_material_id(const vec3<R> &point) const {
  vec3<R> local_point = world_to_local(point);

  // Find which primitive contains this point
  intersect_result_t<R> result = intersect(
      point, vec3<R>(0, 0, 1), std::numeric_limits<R>::max(), HIGH_PRECISION);
  if (result.hit) {
    return result.material_id;
  }

  return 0; // Default material
}

template <typename R>
R voxel_bvh_hybrid<R>::get_material_density(const vec3<R> &point) const {
  uint32_t material_id = get_material_id(point);
  if (material_id < material_densities_.size()) {
    return material_densities_[material_id];
  }
  return static_cast<R>(1.0); // Default density
}

template <typename R>
bool voxel_bvh_hybrid<R>::coarse_intersect_aabb(const vec3<R> &aabb_min,
                                                const vec3<R> &aabb_max) const {
  if (bvh_nodes_.empty())
    return false;

  // Check against root node bounds
  const bvh_node<R> &root = bvh_nodes_[0];

  return !(root.bounds_max.x < aabb_min.x || root.bounds_min.x > aabb_max.x ||
           root.bounds_max.y < aabb_min.y || root.bounds_min.y > aabb_max.y ||
           root.bounds_max.z < aabb_min.z || root.bounds_min.z > aabb_max.z);
}

template <typename R>
bool voxel_bvh_hybrid<R>::simplified_intersect(const vec3<R> &origin,
                                               const vec3<R> &direction) const {
  // Use simplified bounding box test first
  vec3<R> bounds_min, bounds_max;
  get_world_bounds(bounds_min, bounds_max);

  R t_min = 0, t_max = std::numeric_limits<R>::max();
  vec3<R> inv_dir(1.0 / direction.x, 1.0 / direction.y, 1.0 / direction.z);

  for (int i = 0; i < 3; ++i) {
    R t1 = (bounds_min[i] - origin[i]) * inv_dir[i];
    R t2 = (bounds_max[i] - origin[i]) * inv_dir[i];

    t_min = std::max(t_min, std::min(t1, t2));
    t_max = std::min(t_max, std::max(t1, t2));

    if (t_min > t_max)
      return false;
  }

  return true;
}

template <typename R> float voxel_bvh_hybrid<R>::get_performance_cost() const {
  float base_cost = 1.0f;

  // Adjust for precision
  switch (current_precision_) {
  case FAST_APPROXIMATION:
    base_cost *= 0.5f;
    break;
  case BALANCED:
    base_cost *= 1.0f;
    break;
  case HIGH_PRECISION:
    base_cost *= 2.0f;
    break;
  }

  // Adjust for complexity
  switch (current_complexity_) {
  case COARSE:
    base_cost *= 0.3f;
    break;
  case MEDIUM:
    base_cost *= 1.0f;
    break;
  case FINE:
    base_cost *= 2.5f;
    break;
  }

  return base_cost;
}

template <typename R> float voxel_bvh_hybrid<R>::get_accuracy_estimate() const {
  float base_accuracy = 0.95f;

  // Adjust for precision
  switch (current_precision_) {
  case FAST_APPROXIMATION:
    base_accuracy *= 0.8f;
    break;
  case BALANCED:
    base_accuracy *= 0.95f;
    break;
  case HIGH_PRECISION:
    base_accuracy *= 0.99f;
    break;
  }

  // Adjust for complexity
  switch (current_complexity_) {
  case COARSE:
    base_accuracy *= 0.7f;
    break;
  case MEDIUM:
    base_accuracy *= 0.9f;
    break;
  case FINE:
    base_accuracy *= 0.98f;
    break;
  }

  return std::min(base_accuracy, 1.0f);
}

template <typename R>
size_t voxel_bvh_hybrid<R>::estimate_memory_usage() const {
  size_t usage = sizeof(*this);
  usage += bvh_nodes_.size() * sizeof(bvh_node<R>);
  usage += voxel_mask_.mask_data.size() * sizeof(uint8_t);
  usage += vertices_.size() * sizeof(vec3<R>);
  usage += indices_.size() * sizeof(uint32_t);
  usage += material_ids_.size() * sizeof(uint32_t);
  usage += material_densities_.size() * sizeof(R);
  usage += traversal_stack_.capacity() * sizeof(traversal_state<R>);

  if (gpu_ready_) {
    usage += gpu_memory_size_;
  }

  return usage;
}

template <typename R> bool voxel_bvh_hybrid<R>::prepare_for_gpu() {
  if (gpu_ready_)
    return true;

  try {
    // Upload BVH to GPU
    if (!upload_bvh_to_gpu()) {
      return false;
    }

    // Upload voxel mask to GPU
    if (!voxel_mask_.mask_data.empty()) {
#if defined(__CUDACC__)
      cudaError_t err = cudaMalloc(
          &gpu_mask_buffer_, voxel_mask_.mask_data.size() * sizeof(uint8_t));
      if (err != cudaSuccess) {
        cleanup_gpu();
        return false;
      }

      err = cudaMemcpy(gpu_mask_buffer_, voxel_mask_.mask_data.data(),
                       voxel_mask_.mask_data.size() * sizeof(uint8_t),
                       cudaMemcpyHostToDevice);
      if (err != cudaSuccess) {
        cleanup_gpu();
        return false;
      }
#endif
    }

    gpu_ready_ = true;
    return true;
  } catch (...) {
    cleanup_gpu();
    return false;
  }
}

template <typename R> void voxel_bvh_hybrid<R>::cleanup_gpu() {
  if (gpu_bvh_buffer_) {
#if defined(__CUDACC__)
    cudaFree(gpu_bvh_buffer_);
#endif
    gpu_bvh_buffer_ = nullptr;
  }

  if (gpu_mask_buffer_) {
#if defined(__CUDACC__)
    cudaFree(gpu_mask_buffer_);
#endif
    gpu_mask_buffer_ = nullptr;
  }

  gpu_ready_ = false;
  gpu_memory_size_ = 0;
}

template <typename R>
vec3<R> voxel_bvh_hybrid<R>::world_to_local(const vec3<R> &world_point) const {
  vec3<R> local = world_point - transform_origin_;
  local = transform_rotation_ * local;
  local = local / transform_scale_;
  return local;
}

template <typename R>
vec3<R> voxel_bvh_hybrid<R>::local_to_world(const vec3<R> &local_point) const {
  vec3<R> world = local_point * transform_scale_;
  world = transform_rotation_.transpose() * world;
  world = world + transform_origin_;
  return world;
}

template <typename R>
bool voxel_bvh_hybrid<R>::is_inside(const vec3<R> &point) const {
  vec3<R> local_point = world_to_local(point);
  intersect_result_t<R> result = intersect(
      point, vec3<R>(0, 0, 1), std::numeric_limits<R>::max(), HIGH_PRECISION);
  return result.hit && result.dist >= 0;
}

template <typename R>
void voxel_bvh_hybrid<R>::get_world_bounds(vec3<R> &aabb_min,
                                           vec3<R> &aabb_max) const {
  if (!bvh_nodes_.empty()) {
    const bvh_node<R> &root = bvh_nodes_[0];
    aabb_min = local_to_world(root.bounds_min);
    aabb_max = local_to_world(root.bounds_max);
  } else {
    aabb_min = vec3<R>(0);
    aabb_max = vec3<R>(0);
  }
}

template <typename R> void voxel_bvh_hybrid<R>::dump_info() const {
  std::cout << "Voxel-BVH Hybrid Geometry Info:\n";
  std::cout << "  Type: " << static_cast<int>(this->geotype) << "\n";
  std::cout << "  Precision: " << static_cast<int>(current_precision_) << "\n";
  std::cout << "  Complexity: " << static_cast<int>(current_complexity_)
            << "\n";
  std::cout << "  BVH Nodes: " << bvh_nodes_.size() << "\n";
  std::cout << "  Vertices: " << vertices_.size() << "\n";
  std::cout << "  Primitives: " << indices_.size() / 3 << "\n";
  std::cout << "  Voxel Mask: " << voxel_mask_.width << "x"
            << voxel_mask_.height << "\n";
  std::cout << "  GPU Ready: " << (gpu_ready_ ? "Yes" : "No") << "\n";
  std::cout << "  Memory Usage: " << estimate_memory_usage() << " bytes\n";

  bvh_stats stats = get_bvh_stats();
  std::cout << "  Max BVH Depth: " << stats.max_depth << "\n";
  std::cout << "  Avg Primitives/Leaf: " << stats.average_primitives_per_leaf
            << "\n";
}

template <typename R> bool voxel_bvh_hybrid<R>::validate() const {
  if (bvh_nodes_.empty())
    return false;
  if (vertices_.empty())
    return false;
  if (indices_.empty() || indices_.size() % 3 != 0)
    return false;

  return voxel_bvh_utils::validate_bvh(bvh_nodes_);
}

template <typename R>
void voxel_bvh_hybrid<R>::build_bvh(const std::vector<vec3<R>> &vertices,
                                    const std::vector<uint32_t> &indices,
                                    const std::vector<uint32_t> &material_ids) {
  vertices_ = vertices;
  indices_ = indices;
  material_ids_ = material_ids;

  if (indices.empty())
    return;

  // Calculate primitive centroids
  std::vector<vec3<R>> centroids;
  centroids.reserve(indices.size() / 3);

  for (size_t i = 0; i < indices.size(); i += 3) {
    vec3<R> v0 = vertices[indices[i]];
    vec3<R> v1 = vertices[indices[i + 1]];
    vec3<R> v2 = vertices[indices[i + 2]];
    centroids.push_back((v0 + v1 + v2) / static_cast<R>(3.0));
  }

  // Initialize BVH
  uint32_t num_primitives = indices.size() / 3;
  bvh_nodes_.clear();
  bvh_nodes_.reserve(num_primitives * 2); // Reserve space for worst case

  // Create root node
  bvh_nodes_.emplace_back();

  // Build recursively
  std::vector<uint32_t> primitive_indices(num_primitives);
  std::iota(primitive_indices.begin(), primitive_indices.end(), 0);

  build_bvh_recursive(0, 0, num_primitives, centroids, primitive_indices);
}

template <typename R>
void voxel_bvh_hybrid<R>::update_voxel_mask(const vec3<R> &beam_direction,
                                            R distance) {
  voxel_mask_.create_from_geometry(this, beam_direction, distance);
}

template <typename R>
void voxel_bvh_hybrid<R>::set_precision(geometry_precision_t precision) {
  current_precision_ = precision;
  geometry_interface<R>::set_precision(precision);
}

template <typename R>
void voxel_bvh_hybrid<R>::set_complexity(geometry_complexity_t complexity) {
  current_complexity_ = complexity;
  geometry_interface<R>::set_complexity(complexity);
}

template <typename R>
typename voxel_bvh_hybrid<R>::bvh_stats
voxel_bvh_hybrid<R>::get_bvh_stats() const {
  bvh_stats stats{};
  stats.total_nodes = bvh_nodes_.size();
  stats.max_depth = 0;

  uint32_t total_primitives = 0;
  uint32_t leaf_count = 0;

  for (const auto &node : bvh_nodes_) {
    if (node.is_leaf) {
      leaf_count++;
      total_primitives += node.primitive_count;
    }
  }

  stats.leaf_nodes = leaf_count;
  stats.average_primitives_per_leaf =
      leaf_count > 0 ? static_cast<float>(total_primitives) / leaf_count : 0.0f;

  return stats;
}

// Private methods implementation
template <typename R>
void voxel_bvh_hybrid<R>::build_bvh_recursive(
    uint32_t node_id, uint32_t start, uint32_t end,
    std::vector<vec3<R>> &centroids, std::vector<uint32_t> &primitive_indices) {
  if (end - start <= 4) { // Leaf node threshold
    bvh_node<R> &node = bvh_nodes_[node_id];
    node.is_leaf = 1;
    node.primitive_start = start * 3; // Convert to vertex index
    node.primitive_count = (end - start) * 3;
    calculate_node_bounds(node_id, start, end, vertices_, indices_);
    return;
  }

  // Calculate bounds for current node
  calculate_node_bounds(node_id, start, end, vertices_, indices_);

  // Find best split axis and position
  vec3<R> bounds_min = bvh_nodes_[node_id].bounds_min;
  vec3<R> bounds_max = bvh_nodes_[node_id].bounds_max;
  vec3<R> extent = bounds_max - bounds_min;

  int split_axis = 0;
  if (extent.y > extent.x)
    split_axis = 1;
  if (extent.z > extent[split_axis])
    split_axis = 2;

  uint32_t split_pos = find_best_split(start, end, split_axis, centroids);
  if (split_pos == start || split_pos == end) {
    // Cannot split further, make leaf
    bvh_node<R> &node = bvh_nodes_[node_id];
    node.is_leaf = 1;
    node.primitive_start = start * 3;
    node.primitive_count = (end - start) * 3;
    return;
  }

  // Create child nodes
  uint32_t left_child = bvh_nodes_.size();
  uint32_t right_child = bvh_nodes_.size() + 1;

  bvh_nodes_[node_id].left_child = left_child;
  bvh_nodes_[node_id].right_child = right_child;

  bvh_nodes_.emplace_back();
  bvh_nodes_.emplace_back();

  // Recursively build children
  build_bvh_recursive(left_child, start, split_pos, centroids,
                      primitive_indices);
  build_bvh_recursive(right_child, split_pos, end, centroids,
                      primitive_indices);
}

template <typename R>
void voxel_bvh_hybrid<R>::calculate_node_bounds(
    uint32_t node_id, uint32_t start, uint32_t end,
    const std::vector<vec3<R>> &vertices,
    const std::vector<uint32_t> &indices) {
  vec3<R> min_bounds(std::numeric_limits<R>::max());
  vec3<R> max_bounds(std::numeric_limits<R>::lowest());

  for (uint32_t i = start; i < end; ++i) {
    for (int j = 0; j < 3; ++j) {
      vec3<R> vertex = vertices[indices[i * 3 + j]];
      min_bounds.x = std::min(min_bounds.x, vertex.x);
      min_bounds.y = std::min(min_bounds.y, vertex.y);
      min_bounds.z = std::min(min_bounds.z, vertex.z);
      max_bounds.x = std::max(max_bounds.x, vertex.x);
      max_bounds.y = std::max(max_bounds.y, vertex.y);
      max_bounds.z = std::max(max_bounds.z, vertex.z);
    }
  }

  bvh_nodes_[node_id].bounds_min = min_bounds;
  bvh_nodes_[node_id].bounds_max = max_bounds;
}

template <typename R>
uint32_t voxel_bvh_hybrid<R>::find_best_split(
    uint32_t start, uint32_t end, int axis,
    const std::vector<vec3<R>> &centroids) const {
  std::vector<uint32_t> indices(end - start);
  std::iota(indices.begin(), indices.end(), start);

  // Sort by centroid along split axis
  std::sort(indices.begin(), indices.end(),
            [axis, &centroids](uint32_t a, uint32_t b) {
              return centroids[a][axis] < centroids[b][axis];
            });

  // Find best split using simple heuristic (middle)
  return start + (end - start) / 2;
}

template <typename R>
intersect_result_t<R> voxel_bvh_hybrid<R>::traverse_bvh_stackless(
    const vec3<R> &origin, const vec3<R> &direction, R max_distance,
    geometry_precision_t precision) const {
  intersect_result_t<R> result;
  if (bvh_nodes_.empty())
    return result;

  // Simple stackless traversal using while loop
  uint32_t node_stack[64];
  int stack_top = 0;
  node_stack[stack_top++] = 0; // Start with root

  R closest_distance = max_distance;

  while (stack_top > 0) {
    uint32_t node_idx = node_stack[--stack_top];
    const bvh_node<R> &node = bvh_nodes_[node_idx];

    if (!node.intersect_ray(origin, direction, 0, closest_distance)) {
      continue;
    }

    if (node.is_leaf) {
      // Test primitives in this leaf
      for (uint32_t i = 0; i < node.primitive_count; i += 3) {
        intersect_result_t<R> prim_result = intersect_leaf_primitive(
            node.primitive_start + i, origin, direction, max_distance);

        if (prim_result.hit && prim_result.dist < closest_distance &&
            prim_result.dist >= 0) {
          result = prim_result;
          closest_distance = prim_result.dist;
        }
      }
    } else {
      // Add children to stack (order by distance for efficiency)
      if (node.left_child > 0 && stack_top < 63) {
        node_stack[stack_top++] = node.left_child;
      }
      if (node.right_child > 0 && stack_top < 63) {
        node_stack[stack_top++] = node.right_child;
      }
    }
  }

  return result;
}

template <typename R>
intersect_result_t<R> voxel_bvh_hybrid<R>::intersect_leaf_primitive(
    uint32_t primitive_id, const vec3<R> &origin, const vec3<R> &direction,
    R max_distance) const {
  intersect_result_t<R> result;

  if (primitive_id + 2 >= indices_.size())
    return result;

  // Get triangle vertices
  vec3<R> v0 = vertices_[indices_[primitive_id]];
  vec3<R> v1 = vertices_[indices_[primitive_id + 1]];
  vec3<R> v2 = vertices_[indices_[primitive_id + 2]];

  // Möller-Trumbore ray-triangle intersection
  vec3<R> edge1 = v1 - v0;
  vec3<R> edge2 = v2 - v0;
  vec3<R> h = cross(direction, edge2);
  R a = dot(edge1, h);

  if (std::abs(a) < 1e-6)
    return result; // Ray parallel to triangle

  R f = 1.0 / a;
  vec3<R> s = origin - v0;
  R u = f * dot(s, h);

  if (u < 0.0 || u > 1.0)
    return result;

  vec3<R> q = cross(s, edge1);
  R v = f * dot(direction, q);

  if (v < 0.0 || u + v > 1.0)
    return result;

  R t = f * dot(edge2, q);

  if (t > 1e-6 && t < max_distance) {
    result.hit = true;
    result.dist = t;
    result.normal = normalize(cross(edge1, edge2));
    result.material_id = (primitive_id / 3) < material_ids_.size()
                             ? material_ids_[primitive_id / 3]
                             : 0;
  }

  return result;
}

template <typename R> bool voxel_bvh_hybrid<R>::upload_bvh_to_gpu() {
#if defined(__CUDACC__)
  if (bvh_nodes_.empty())
    return false;

  size_t buffer_size = bvh_nodes_.size() * sizeof(bvh_node<R>);
  cudaError_t err = cudaMalloc(&gpu_bvh_buffer_, buffer_size);
  if (err != cudaSuccess)
    return false;

  err = cudaMemcpy(gpu_bvh_buffer_, bvh_nodes_.data(), buffer_size,
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cudaFree(gpu_bvh_buffer_);
    gpu_bvh_buffer_ = nullptr;
    return false;
  }

  gpu_memory_size_ += buffer_size;
  return true;
#else
  return false; // CUDA not available
#endif
}

// voxel_mask implementation
template <typename R>
void voxel_mask<R>::create_from_geometry(const geometry_interface<R> *geometry,
                                         const vec3<R> &beam_direction,
                                         R distance, uint16_t resolution) {
  width = height = resolution;
  world_width = world_height = distance * 2; // Simple assumption
  origin = vec3<R>(-world_width / 2, -world_height / 2, 0);

  mask_data.resize(width * height, 0);

  // Simple ray casting to create mask
  for (uint16_t y = 0; y < height; ++y) {
    for (uint16_t x = 0; x < width; ++x) {
      R fx = (x / static_cast<R>(width)) * world_width + origin.x;
      R fy = (y / static_cast<R>(height)) * world_height + origin.y;

      vec3<R> ray_origin(fx, fy, -distance);
      vec3<R> ray_direction = beam_direction;

      intersect_result_t<R> result =
          geometry->intersect(ray_origin, ray_direction, distance * 2);
      if (result.hit) {
        mask_data[y * width + x] = 1;
      }
    }
  }
}

// Utility functions implementation
namespace voxel_bvh_utils {
template <typename R> bool validate_bvh(const std::vector<bvh_node<R>> &nodes) {
  if (nodes.empty())
    return false;

  // Basic validation - check root exists and has valid bounds
  const bvh_node<R> &root = nodes[0];
  if (root.bounds_min.x > root.bounds_max.x ||
      root.bounds_min.y > root.bounds_max.y ||
      root.bounds_min.z > root.bounds_max.z) {
    return false;
  }

  return true;
}
} // namespace voxel_bvh_utils

// Explicit template instantiations
template class voxel_bvh_hybrid<float>;
template class voxel_bvh_hybrid<double>;
template struct voxel_mask<float>;
template struct voxel_mask<double>;

} // namespace mqi
