#ifndef MQI_PHASE4_PLUGIN_SYSTEM_HPP
#define MQI_PHASE4_PLUGIN_SYSTEM_HPP

/// \file
///
/// Phase 4: Plugin Architecture for Runtime-Selectable Geometry Modules
///
/// This header implements a plugin system that allows runtime selection and
/// loading of different geometry modules with performance/accuracy trade-offs.

#include "mqi_common.hpp"
#include "mqi_geometry_interface.hpp"
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace mqi {

/// Plugin metadata for registration and discovery
struct plugin_metadata {
  std::string name;             ///< Plugin name
  std::string version;          ///< Plugin version
  std::string description;      ///< Plugin description
  geometry_type supported_type; ///< Geometry type this plugin supports
  std::vector<geometry_precision_t>
      supported_precisions; ///< Supported precision levels
  std::vector<geometry_complexity_t>
      supported_complexities;  ///< Supported complexity levels
  float min_performance_cost;  ///< Minimum performance cost
  float max_performance_cost;  ///< Maximum performance cost
  float min_accuracy_estimate; ///< Minimum accuracy estimate
  float max_accuracy_estimate; ///< Maximum accuracy estimate
  size_t min_memory_usage;     ///< Minimum memory usage estimate
  size_t max_memory_usage;     ///< Maximum memory usage estimate
  bool requires_gpu;           ///< Whether plugin requires GPU support

  plugin_metadata()
      : version("1.0"), supported_type(UNKNOWN1), min_performance_cost(0.1f),
        max_performance_cost(10.0f), min_accuracy_estimate(0.1f),
        max_accuracy_estimate(1.0f), min_memory_usage(1024),
        max_memory_usage(1024 * 1024 * 1024), requires_gpu(false) {}
};

/// Plugin creation function signature
template <typename R>
using geometry_creator_t = std::function<std::unique_ptr<geometry_interface<R>>(
    geometry_precision_t, geometry_complexity_t)>;

/// Plugin validation function signature
template <typename R>
using plugin_validator_t = std::function<bool(const geometry_interface<R> *)>;

/// Plugin registry for managing available geometry plugins
template <typename R> class geometry_plugin_registry {
private:
  std::unordered_map<std::string, plugin_metadata> metadata_map_;
  std::unordered_map<std::string, geometry_creator_t<R>> creator_map_;
  std::unordered_map<std::string, plugin_validator_t<R>> validator_map_;
  mutable std::mutex registry_mutex_;

public:
  /// Singleton access
  static geometry_plugin_registry &instance() {
    static geometry_plugin_registry registry;
    return registry;
  }

  /// Register a geometry plugin
  /// \param metadata Plugin metadata
  /// \param creator Function to create plugin instance
  /// \param validator Optional validation function
  /// \return True if registration successful
  bool register_plugin(const plugin_metadata &metadata,
                       geometry_creator_t<R> creator,
                       plugin_validator_t<R> validator = nullptr);

  /// Unregister a plugin by name
  /// \param name Plugin name
  /// \return True if unregistration successful
  bool unregister_plugin(const std::string &name);

  /// Check if plugin is registered
  /// \param name Plugin name
  /// \return True if registered
  bool is_plugin_registered(const std::string &name) const;

  /// Create geometry instance by plugin name
  /// \param name Plugin name
  /// \param precision Precision level
  /// \param complexity Complexity level
  /// \return Unique pointer to geometry instance
  std::unique_ptr<geometry_interface<R>>
  create_geometry(const std::string &name,
                  geometry_precision_t precision = BALANCED,
                  geometry_complexity_t complexity = MEDIUM) const;

  /// Get plugin metadata
  /// \param name Plugin name
  /// \return Plugin metadata (empty if not found)
  plugin_metadata get_plugin_metadata(const std::string &name) const;

  /// Get list of all registered plugins
  /// \return Vector of plugin names
  std::vector<std::string> get_registered_plugins() const;

  /// Get plugins that support specific geometry type
  /// \param type Geometry type
  /// \return Vector of plugin names
  std::vector<std::string> get_plugins_for_type(geometry_type type) const;

  /// Get plugins that support specific precision
  /// \param precision Precision level
  /// \return Vector of plugin names
  std::vector<std::string>
  get_plugins_for_precision(geometry_precision_t precision) const;

  /// Validate plugin instance
  /// \param name Plugin name
  /// \param geometry Geometry instance to validate
  /// \return True if valid
  bool validate_plugin(const std::string &name,
                       const geometry_interface<R> *geometry) const;

  /// Find best plugin for given requirements
  /// \param type Geometry type
  /// \param max_performance_cost Maximum acceptable performance cost
  /// \param min_accuracy Minimum acceptable accuracy
  /// \param requires_gpu Whether GPU support is required
  /// \return Best plugin name (empty if none found)
  std::string find_best_plugin(
      geometry_type type,
      float max_performance_cost = std::numeric_limits<float>::max(),
      float min_accuracy = 0.0f, bool requires_gpu = false) const;

private:
  geometry_plugin_registry() = default;
  ~geometry_plugin_registry() = default;
  geometry_plugin_registry(const geometry_plugin_registry &) = delete;
  geometry_plugin_registry &
  operator=(const geometry_plugin_registry &) = delete;
};

/// Plugin loader for dynamic loading of geometry plugins
template <typename R> class geometry_plugin_loader {
public:
  /// Load plugin from shared library
  /// \param library_path Path to shared library
  /// \param init_function Name of initialization function
  /// \return True if loading successful
  bool load_plugin(const std::string &library_path,
                   const std::string &init_function = "init_plugin");

  /// Unload plugin
  /// \param library_path Path to shared library
  /// \return True if unloading successful
  bool unload_plugin(const std::string &library_path);

  /// Get list of loaded libraries
  /// \return Vector of library paths
  std::vector<std::string> get_loaded_libraries() const;

  /// Check if library is loaded
  /// \param library_path Path to shared library
  /// \return True if loaded
  bool is_library_loaded(const std::string &library_path) const;

private:
  std::unordered_map<std::string, void *> loaded_libraries_;
  mutable std::mutex loader_mutex_;
};

/// Plugin manager that coordinates registry and loader
template <typename R> class geometry_plugin_manager {
private:
  geometry_plugin_registry<R> &registry_;
  geometry_plugin_loader<R> loader_;

public:
  /// Constructor
  geometry_plugin_manager()
      : registry_(geometry_plugin_registry<R>::instance()) {}

  /// Register built-in plugins
  void register_builtin_plugins();

  /// Load and register plugins from directory
  /// \param plugin_dir Directory containing plugin libraries
  /// \return Number of plugins loaded
  size_t load_plugins_from_directory(const std::string &plugin_dir);

  /// Create geometry instance with automatic plugin selection
  /// \param type Geometry type
  /// \param precision Precision level
  /// \param complexity Complexity level
  /// \param max_performance_cost Maximum acceptable cost
  /// \return Unique pointer to geometry instance
  std::unique_ptr<geometry_interface<R>> create_optimal_geometry(
      geometry_type type, geometry_precision_t precision = BALANCED,
      geometry_complexity_t complexity = MEDIUM,
      float max_performance_cost = std::numeric_limits<float>::max());

  /// Create geometry instance with specific plugin
  /// \param plugin_name Plugin name
  /// \param precision Precision level
  /// \param complexity Complexity level
  /// @return Unique pointer to geometry instance
  std::unique_ptr<geometry_interface<R>>
  create_geometry(const std::string &plugin_name,
                  geometry_precision_t precision = BALANCED,
                  geometry_complexity_t complexity = MEDIUM);

  /// Get performance benchmark for all plugins
  /// \return Map of plugin names to performance metrics
  std::unordered_map<std::string, float>
  benchmark_plugins(geometry_type type, size_t num_iterations = 1000);

  /// Get registry access
  geometry_plugin_registry<R> &get_registry() { return registry_; }

  /// Get loader access
  geometry_plugin_loader<R> &get_loader() { return loader_; }
};

/// Plugin registration helper macro
#define MQI_REGISTER_GEOMETRY_PLUGIN(R, name, metadata)                        \
  extern "C" {                                                                 \
  std::unique_ptr<mqi::geometry_interface<R>>                                  \
      create_##name##_plugin(mqi::geometry_precision_t precision,              \
                             mqi::geometry_complexity_t complexity);           \
  bool validate_##name##_plugin(const mqi::geometry_interface<R> *geometry);   \
                                                                               \
  void init_##name##_plugin() {                                                \
    mqi::geometry_plugin_registry<R>::instance().register_plugin(              \
        metadata, create_##name##_plugin, validate_##name##_plugin);           \
  }                                                                            \
  }

/// Built-in plugin creators
namespace builtin_plugins {
/// Create voxel-BVH hybrid plugin
template <typename R>
std::unique_ptr<geometry_interface<R>>
create_voxel_bvh_plugin(geometry_precision_t precision,
                        geometry_complexity_t complexity);

/// Create grid-based plugin
template <typename R>
std::unique_ptr<geometry_interface<R>>
create_grid_plugin(geometry_precision_t precision,
                   geometry_complexity_t complexity);

/// Create simple AABB plugin for testing
template <typename R>
std::unique_ptr<geometry_interface<R>>
create_aabb_plugin(geometry_precision_t precision,
                   geometry_complexity_t complexity);

/// Register all built-in plugins
template <typename R> void register_all_builtin_plugins();
} // namespace builtin_plugins

/// Plugin performance profiler
template <typename R> class plugin_profiler {
public:
  struct profile_result {
    std::string plugin_name;
    geometry_precision_t precision;
    geometry_complexity_t complexity;
    float avg_intersection_time_ms;
    float min_intersection_time_ms;
    float max_intersection_time_ms;
    float avg_accuracy;
    float memory_usage_mb;
    size_t num_intersections;
  };

  /// Profile plugin performance
  /// \param geometry Geometry instance to profile
  /// \param plugin_name Plugin name for reporting
  /// \param num_tests Number of test rays
  /// \return Profile results
  profile_result profile_plugin(const geometry_interface<R> *geometry,
                                const std::string &plugin_name,
                                size_t num_tests = 10000);

  /// Compare multiple plugins
  /// \param geometries Map of plugin names to geometry instances
  /// \param num_tests Number of test rays per plugin
  /// @return Vector of profile results
  std::vector<profile_result> compare_plugins(
      const std::unordered_map<
          std::string, std::unique_ptr<geometry_interface<R>>> &geometries,
      size_t num_tests = 10000);

private:
  std::vector<vec3<R>> generate_test_rays(size_t num_rays,
                                          const vec3<R> &bounds_min,
                                          const vec3<R> &bounds_max) const;
};

} // namespace mqi

#endif // MQI_PHASE4_PLUGIN_SYSTEM_HPP
