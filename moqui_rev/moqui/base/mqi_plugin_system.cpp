#include "mqi_plugin_system.hpp"
#include "mqi_voxel_bvh.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <random>

#if defined(__linux__)
#include <dlfcn.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

namespace mqi {

// geometry_plugin_registry implementation
template <typename R>
bool geometry_plugin_registry<R>::register_plugin(
    const plugin_metadata &metadata, geometry_creator_t<R> creator,
    plugin_validator_t<R> validator) {
  std::lock_guard<std::mutex> lock(registry_mutex_);

  if (metadata.name.empty() || !creator) {
    return false;
  }

  metadata_map_[metadata.name] = metadata;
  creator_map_[metadata.name] = creator;
  if (validator) {
    validator_map_[metadata.name] = validator;
  }

  return true;
}

template <typename R>
bool geometry_plugin_registry<R>::unregister_plugin(const std::string &name) {
  std::lock_guard<std::mutex> lock(registry_mutex_);

  size_t erased = metadata_map_.erase(name);
  creator_map_.erase(name);
  validator_map_.erase(name);

  return erased > 0;
}

template <typename R>
bool geometry_plugin_registry<R>::is_plugin_registered(
    const std::string &name) const {
  std::lock_guard<std::mutex> lock(registry_mutex_);
  return metadata_map_.find(name) != metadata_map_.end();
}

template <typename R>
std::unique_ptr<geometry_interface<R>>
geometry_plugin_registry<R>::create_geometry(
    const std::string &name, geometry_precision_t precision,
    geometry_complexity_t complexity) const {
  std::lock_guard<std::mutex> lock(registry_mutex_);

  auto it = creator_map_.find(name);
  if (it != creator_map_.end()) {
    return it->second(precision, complexity);
  }

  return nullptr;
}

template <typename R>
plugin_metadata geometry_plugin_registry<R>::get_plugin_metadata(
    const std::string &name) const {
  std::lock_guard<std::mutex> lock(registry_mutex_);

  auto it = metadata_map_.find(name);
  if (it != metadata_map_.end()) {
    return it->second;
  }

  return plugin_metadata();
}

template <typename R>
std::vector<std::string>
geometry_plugin_registry<R>::get_registered_plugins() const {
  std::lock_guard<std::mutex> lock(registry_mutex_);

  std::vector<std::string> plugins;
  plugins.reserve(metadata_map_.size());

  for (const auto &pair : metadata_map_) {
    plugins.push_back(pair.first);
  }

  return plugins;
}

template <typename R>
std::vector<std::string>
geometry_plugin_registry<R>::get_plugins_for_type(geometry_type type) const {
  std::lock_guard<std::mutex> lock(registry_mutex_);

  std::vector<std::string> matching_plugins;

  for (const auto &pair : metadata_map_) {
    if (pair.second.supported_type == type) {
      matching_plugins.push_back(pair.first);
    }
  }

  return matching_plugins;
}

template <typename R>
std::vector<std::string> geometry_plugin_registry<R>::get_plugins_for_precision(
    geometry_precision_t precision) const {
  std::lock_guard<std::mutex> lock(registry_mutex_);

  std::vector<std::string> matching_plugins;

  for (const auto &pair : metadata_map_) {
    const auto &supported_precisions = pair.second.supported_precisions;
    if (std::find(supported_precisions.begin(), supported_precisions.end(),
                  precision) != supported_precisions.end()) {
      matching_plugins.push_back(pair.first);
    }
  }

  return matching_plugins;
}

template <typename R>
bool geometry_plugin_registry<R>::validate_plugin(
    const std::string &name, const geometry_interface<R> *geometry) const {
  std::lock_guard<std::mutex> lock(registry_mutex_);

  auto it = validator_map_.find(name);
  if (it != validator_map_.end() && geometry) {
    return it->second(geometry);
  }

  return geometry ? geometry->validate() : false;
}

template <typename R>
std::string geometry_plugin_registry<R>::find_best_plugin(
    geometry_type type, float max_performance_cost, float min_accuracy,
    bool requires_gpu) const {
  std::lock_guard<std::mutex> lock(registry_mutex_);

  std::string best_plugin;
  float best_score = -1.0f;

  for (const auto &pair : metadata_map_) {
    const auto &metadata = pair.second;

    if (metadata.supported_type != type)
      continue;
    if (metadata.max_performance_cost > max_performance_cost)
      continue;
    if (metadata.max_accuracy_estimate < min_accuracy)
      continue;
    if (requires_gpu && !metadata.requires_gpu)
      continue;

    // Calculate score based on accuracy vs performance trade-off
    float accuracy_score =
        (metadata.max_accuracy_estimate + metadata.min_accuracy_estimate) /
        2.0f;
    float performance_score =
        1.0f /
        ((metadata.max_performance_cost + metadata.min_performance_cost) /
         2.0f);
    float combined_score = accuracy_score * 0.6f + performance_score * 0.4f;

    if (combined_score > best_score) {
      best_score = combined_score;
      best_plugin = pair.first;
    }
  }

  return best_plugin;
}

// geometry_plugin_loader implementation
template <typename R>
bool geometry_plugin_loader<R>::load_plugin(const std::string &library_path,
                                            const std::string &init_function) {
  std::lock_guard<std::mutex> lock(loader_mutex_);

  if (is_library_loaded(library_path)) {
    return true; // Already loaded
  }

#if defined(__linux__)
  void *handle = dlopen(library_path.c_str(), RTLD_LAZY);
  if (!handle) {
    std::cerr << "Failed to load library " << library_path << ": " << dlerror()
              << std::endl;
    return false;
  }

  // Get initialization function
  typedef void (*init_func_t)();
  init_func_t init_func = (init_func_t)dlsym(handle, init_function.c_str());

  if (!init_func) {
    std::cerr << "Failed to find init function " << init_function << " in "
              << library_path << std::endl;
    dlclose(handle);
    return false;
  }

  // Call initialization function
  init_func();

  loaded_libraries_[library_path] = handle;
  return true;

#elif defined(_WIN32)
  HMODULE handle = LoadLibraryA(library_path.c_str());
  if (!handle) {
    std::cerr << "Failed to load library " << library_path << std::endl;
    return false;
  }

  typedef void (*init_func_t)();
  init_func_t init_func =
      (init_func_t)GetProcAddress(handle, init_function.c_str());

  if (!init_func) {
    std::cerr << "Failed to find init function " << init_function << " in "
              << library_path << std::endl;
    FreeLibrary(handle);
    return false;
  }

  init_func();
  loaded_libraries_[library_path] = handle;
  return true;

#else
  std::cerr << "Dynamic loading not supported on this platform" << std::endl;
  return false;
#endif
}

template <typename R>
bool geometry_plugin_loader<R>::unload_plugin(const std::string &library_path) {
  std::lock_guard<std::mutex> lock(loader_mutex_);

  auto it = loaded_libraries_.find(library_path);
  if (it == loaded_libraries_.end()) {
    return false; // Not loaded
  }

#if defined(__linux__)
  dlclose(it->second);
#elif defined(_WIN32)
  FreeLibrary((HMODULE)it->second);
#endif

  loaded_libraries_.erase(it);
  return true;
}

template <typename R>
std::vector<std::string>
geometry_plugin_loader<R>::get_loaded_libraries() const {
  std::lock_guard<std::mutex> lock(loader_mutex_);

  std::vector<std::string> libraries;
  libraries.reserve(loaded_libraries_.size());

  for (const auto &pair : loaded_libraries_) {
    libraries.push_back(pair.first);
  }

  return libraries;
}

template <typename R>
bool geometry_plugin_loader<R>::is_library_loaded(
    const std::string &library_path) const {
  std::lock_guard<std::mutex> lock(loader_mutex_);
  return loaded_libraries_.find(library_path) != loaded_libraries_.end();
}

// geometry_plugin_manager implementation
template <typename R>
void geometry_plugin_manager<R>::register_builtin_plugins() {
  builtin_plugins::register_all_builtin_plugins<R>();
}

template <typename R>
size_t geometry_plugin_manager<R>::load_plugins_from_directory(
    const std::string &plugin_dir) {
  size_t loaded_count = 0;

  try {
    for (const auto &entry : std::filesystem::directory_iterator(plugin_dir)) {
      if (entry.is_regular_file()) {
        std::string path = entry.path().string();
        std::string extension = entry.path().extension().string();

#if defined(__linux__)
        if (extension == ".so") {
#elif defined(_WIN32)
        if (extension == ".dll") {
#else
        continue;
#endif
          if (loader_.load_plugin(path)) {
            loaded_count++;
          }
        }
      }
    }
  } catch (const std::filesystem::filesystem_error &e) {
    std::cerr << "Error loading plugins from directory " << plugin_dir << ": "
              << e.what() << std::endl;
  }

  return loaded_count;
}

template <typename R>
std::unique_ptr<geometry_interface<R>>
geometry_plugin_manager<R>::create_optimal_geometry(
    geometry_type type, geometry_precision_t precision,
    geometry_complexity_t complexity, float max_performance_cost) {
  std::string best_plugin =
      registry_.find_best_plugin(type, max_performance_cost);
  if (best_plugin.empty()) {
    return nullptr;
  }

  return registry_.create_geometry(best_plugin, precision, complexity);
}

template <typename R>
std::unique_ptr<geometry_interface<R>>
geometry_plugin_manager<R>::create_geometry(const std::string &plugin_name,
                                            geometry_precision_t precision,
                                            geometry_complexity_t complexity) {
  return registry_.create_geometry(plugin_name, precision, complexity);
}

template <typename R>
std::unordered_map<std::string, float>
geometry_plugin_manager<R>::benchmark_plugins(geometry_type type,
                                              size_t num_iterations) {
  std::unordered_map<std::string, float> results;
  plugin_profiler<R> profiler;

  auto plugins = registry_.get_plugins_for_type(type);

  for (const auto &plugin_name : plugins) {
    auto geometry = registry_.create_geometry(plugin_name);
    if (geometry) {
      auto profile =
          profiler.profile_plugin(geometry.get(), plugin_name, num_iterations);
      results[plugin_name] = profile.avg_intersection_time_ms;
    }
  }

  return results;
}

// Built-in plugin implementations
namespace builtin_plugins {
template <typename R>
std::unique_ptr<geometry_interface<R>>
create_voxel_bvh_plugin(geometry_precision_t precision,
                        geometry_complexity_t complexity) {
  auto geometry =
      std::make_unique<voxel_bvh_hybrid<R>>(PATIENT, precision, complexity);
  return std::move(geometry);
}

template <typename R>
bool validate_voxel_bvh_plugin(const geometry_interface<R> *geometry) {
  return geometry ? geometry->validate() : false;
}

template <typename R>
std::unique_ptr<geometry_interface<R>>
create_grid_plugin(geometry_precision_t precision,
                   geometry_complexity_t complexity) {
  // This would be a grid-based implementation
  // For now, return nullptr as placeholder
  return nullptr;
}

template <typename R>
bool validate_grid_plugin(const geometry_interface<R> *geometry) {
  return geometry ? geometry->validate() : false;
}

template <typename R>
std::unique_ptr<geometry_interface<R>>
create_aabb_plugin(geometry_precision_t precision,
                   geometry_complexity_t complexity) {
  // This would be a simple AABB-based implementation for testing
  // For now, return nullptr as placeholder
  return nullptr;
}

template <typename R>
bool validate_aabb_plugin(const geometry_interface<R> *geometry) {
  return geometry ? geometry->validate() : false;
}

template <typename R> void register_all_builtin_plugins() {
  auto &registry = geometry_plugin_registry<R>::instance();

  // Register Voxel-BVH plugin
  {
    plugin_metadata metadata;
    metadata.name = "voxel_bvh_hybrid";
    metadata.description =
        "Voxel-BVH hybrid system with coarse-to-fine filtering";
    metadata.supported_type = PATIENT;
    metadata.supported_precisions = {FAST_APPROXIMATION, BALANCED,
                                     HIGH_PRECISION};
    metadata.supported_complexities = {COARSE, MEDIUM, FINE};
    metadata.min_performance_cost = 0.5f;
    metadata.max_performance_cost = 3.0f;
    metadata.min_accuracy_estimate = 0.7f;
    metadata.max_accuracy_estimate = 0.99f;
    metadata.min_memory_usage = 1024 * 100;        // 100KB
    metadata.max_memory_usage = 1024 * 1024 * 100; // 100MB
    metadata.requires_gpu = true;

    registry.register_plugin(metadata, create_voxel_bvh_plugin<R>,
                             validate_voxel_bvh_plugin<R>);
  }

  // Register Grid plugin (placeholder)
  {
    plugin_metadata metadata;
    metadata.name = "grid_based";
    metadata.description = "Traditional grid-based geometry system";
    metadata.supported_type = PATIENT;
    metadata.supported_precisions = {BALANCED};
    metadata.supported_complexities = {MEDIUM};
    metadata.min_performance_cost = 1.0f;
    metadata.max_performance_cost = 2.0f;
    metadata.min_accuracy_estimate = 0.8f;
    metadata.max_accuracy_estimate = 0.9f;
    metadata.min_memory_usage = 1024 * 50;        // 50KB
    metadata.max_memory_usage = 1024 * 1024 * 50; // 50MB
    metadata.requires_gpu = false;

    registry.register_plugin(metadata, create_grid_plugin<R>,
                             validate_grid_plugin<R>);
  }

  // Register AABB plugin (placeholder)
  {
    plugin_metadata metadata;
    metadata.name = "simple_aabb";
    metadata.description = "Simple AABB-based geometry for testing";
    metadata.supported_type = PATIENT;
    metadata.supported_precisions = {FAST_APPROXIMATION};
    metadata.supported_complexities = {COARSE};
    metadata.min_performance_cost = 0.1f;
    metadata.max_performance_cost = 0.5f;
    metadata.min_accuracy_estimate = 0.5f;
    metadata.max_accuracy_estimate = 0.7f;
    metadata.min_memory_usage = 1024 * 10;  // 10KB
    metadata.max_memory_usage = 1024 * 100; // 100KB
    metadata.requires_gpu = false;

    registry.register_plugin(metadata, create_aabb_plugin<R>,
                             validate_aabb_plugin<R>);
  }
}
} // namespace builtin_plugins

// plugin_profiler implementation
template <typename R>
typename plugin_profiler<R>::profile_result
plugin_profiler<R>::profile_plugin(const geometry_interface<R> *geometry,
                                   const std::string &plugin_name,
                                   size_t num_tests) {
  profile_result result;
  result.plugin_name = plugin_name;
  result.precision = geometry->precision;
  result.complexity = geometry->complexity;
  result.num_intersections = num_tests;

  // Get geometry bounds for test ray generation
  vec3<R> bounds_min, bounds_max;
  geometry->get_world_bounds(bounds_min, bounds_max);

  // Generate test rays
  auto test_rays = generate_test_rays(num_tests * 2, bounds_min, bounds_max);

  // Profile intersection performance
  auto start_time = std::chrono::high_resolution_clock::now();

  float total_time = 0.0f;
  float min_time = std::numeric_limits<float>::max();
  float max_time = 0.0f;
  size_t successful_intersections = 0;

  for (size_t i = 0; i < num_tests; ++i) {
    vec3<R> origin = test_rays[i];
    vec3<R> direction = test_rays[num_tests + i];

    auto ray_start = std::chrono::high_resolution_clock::now();
    intersect_result_t<R> intersection = geometry->intersect(origin, direction);
    auto ray_end = std::chrono::high_resolution_clock::now();

    float ray_time =
        std::chrono::duration<float, std::milli>(ray_end - ray_start).count();

    total_time += ray_time;
    min_time = std::min(min_time, ray_time);
    max_time = std::max(max_time, ray_time);

    if (intersection.hit) {
      successful_intersections++;
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto total_duration =
      std::chrono::duration<float, std::milli>(end_time - start_time);

  result.avg_intersection_time_ms = total_time / num_tests;
  result.min_intersection_time_ms = min_time;
  result.max_intersection_time_ms = max_time;
  result.avg_accuracy =
      static_cast<float>(successful_intersections) / num_tests;
  result.memory_usage_mb =
      static_cast<float>(geometry->estimate_memory_usage()) /
      (1024.0f * 1024.0f);

  return result;
}

template <typename R>
std::vector<typename plugin_profiler<R>::profile_result>
plugin_profiler<R>::compare_plugins(
    const std::unordered_map<
        std::string, std::unique_ptr<geometry_interface<R>>> &geometries,
    size_t num_tests) {
  std::vector<profile_result> results;

  for (const auto &pair : geometries) {
    if (pair.second) {
      results.push_back(
          profile_plugin(pair.second.get(), pair.first, num_tests));
    }
  }

  // Sort by average performance
  std::sort(results.begin(), results.end(),
            [](const profile_result &a, const profile_result &b) {
              return a.avg_intersection_time_ms < b.avg_intersection_time_ms;
            });

  return results;
}

template <typename R>
std::vector<vec3<R>>
plugin_profiler<R>::generate_test_rays(size_t num_rays,
                                       const vec3<R> &bounds_min,
                                       const vec3<R> &bounds_max) const {
  std::vector<vec3<R>> rays;
  rays.reserve(num_rays * 2);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<R> dist_x(bounds_min.x, bounds_max.x);
  std::uniform_real_distribution<R> dist_y(bounds_min.y, bounds_max.y);
  std::uniform_real_distribution<R> dist_z(bounds_min.z, bounds_max.z);

  for (size_t i = 0; i < num_rays; ++i) {
    // Generate random origin
    vec3<R> origin(dist_x(gen), dist_y(gen), dist_z(gen));
    rays.push_back(origin);

    // Generate random direction (normalized)
    vec3<R> direction(dist_x(gen), dist_y(gen), dist_z(gen));
    direction = (direction - origin);
    direction.normalize();
    rays.push_back(direction);
  }

  return rays;
}

// Explicit template instantiations
template class geometry_plugin_registry<float>;
template class geometry_plugin_registry<double>;
template class geometry_plugin_loader<float>;
template class geometry_plugin_loader<double>;
template class geometry_plugin_manager<float>;
template class geometry_plugin_manager<double>;
template class plugin_profiler<float>;
template class plugin_profiler<double>;

} // namespace mqi
