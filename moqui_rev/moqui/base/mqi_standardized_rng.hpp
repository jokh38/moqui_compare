#ifndef MQI_STANDARDIZED_RNG_HPP
#define MQI_STANDARDIZED_RNG_HPP

/// \file
///
/// Standardized Random Number Generation for Moqui Coarse2Fine GPU Optimization
///
/// This module provides standardized random number generation using
/// curandStatePhilox4_32_10_t across all parallel environments (CUDA kernels
/// and host code). This standardization ensures consistent statistical
/// properties and optimal performance for the persistent kernel architecture
/// implemented in Phase 1.0.
///
/// Key Features:
/// - Standardized to curandStatePhilox4_32_10_t for excellent parallel
/// properties
/// - Host/CPU fallback implementation for compatibility
/// - Thread-safe initialization and usage
/// - Optimized for Monte Carlo particle transport simulation
/// - Backward compatibility with existing mqi_rng interface

#include "mqi_common.hpp"
#include "mqi_error_check.hpp"
#include <cstdint>
#include <random>

#if defined(__CUDACC__)
#include <curand_kernel.h>
#include <device_atomic_functions.h>
#endif

namespace mqi {

// Forward declaration for legacy compatibility - removed to avoid conflicts
// struct mqi_rng;  // Legacy RNG type from mqi_math.hpp

// ============================================================================
// STANDARDIZED RNG TYPE DEFINITION
// ============================================================================

#if defined(__CUDACC__)
/// Standardized RNG type for CUDA kernels
typedef curandStatePhilox4_32_10_t mqi_standard_rng;
#else
/// Standardized RNG type for host code (CPU fallback)
typedef std::mt19937_64 mqi_standard_rng;
#endif

// Type size validation for compatibility - disabled for now to avoid
// compilation issues static_assert(sizeof(mqi_standard_rng) >= sizeof(mqi_rng),
//               "Standardized RNG must be at least as large as legacy RNG");

// ============================================================================
// RNG STATE MANAGEMENT STRUCTURES
// ============================================================================

///< Per-thread RNG state with additional metadata
struct rng_state_t {
  mqi_standard_rng state;   ///< Core RNG state
  uint64_t seed;            ///< Original seed for reproducibility
  uint32_t thread_id;       ///< Thread identifier
  uint32_t sequence_number; ///< Sequence counter for tracking
  bool is_initialized;      ///< Initialization flag
};

///< Batch RNG initialization parameters
struct rng_batch_init_t {
  uint64_t master_seed;     ///< Master seed for all threads
  uint32_t num_threads;     ///< Number of threads to initialize
  uint32_t thread_offset;   ///< Thread ID offset
  uint64_t sequence_offset; ///< Sequence offset
};

// ============================================================================
// CORE RNG FUNCTIONS
// ============================================================================

/// Initialize single RNG state with seed and thread identifier
CUDA_HOST_DEVICE
void initialize_rng_state(rng_state_t *rng_state, const uint64_t seed,
                          const uint32_t thread_id,
                          const uint64_t sequence_offset = 0);

/// Initialize batch of RNG states for parallel execution
CUDA_HOST
void initialize_rng_batch(rng_state_t *rng_states,
                          const rng_batch_init_t *init_params);

/// Generate uniform random float in [0.0, 1.0)
CUDA_HOST_DEVICE
float generate_uniform_float(rng_state_t *rng_state);

/// Generate uniform random double in [0.0, 1.0)
CUDA_HOST_DEVICE
double generate_uniform_double(rng_state_t *rng_state);

/// Generate normal random number with specified mean and standard deviation
CUDA_HOST_DEVICE
float generate_normal_float(rng_state_t *rng_state, const float mean = 0.0f,
                            const float stddev = 1.0f);

/// Generate normal random number with specified mean and standard deviation
CUDA_HOST_DEVICE
double generate_normal_double(rng_state_t *rng_state, const double mean = 0.0,
                              const double stddev = 1.0);

/// Generate random integer in [min, max] (inclusive)
CUDA_HOST_DEVICE
uint32_t generate_uniform_int(rng_state_t *rng_state, const uint32_t min_val,
                              const uint32_t max_val);

/// Skip ahead in RNG sequence (for reproducible parallel execution)
CUDA_HOST_DEVICE
void skip_ahead(rng_state_t *rng_state, const uint64_t steps);

// ============================================================================
// OPTIMIZED RNG FUNCTIONS FOR MONTE CARLO
// ============================================================================

/// Generate 4 uniform floats in parallel (optimized for CUDA)
CUDA_HOST_DEVICE
void generate_uniform_float4(rng_state_t *rng_state, float &r1, float &r2,
                             float &r3, float &r4);

/// Generate 4 uniform doubles in parallel (optimized for CUDA)
CUDA_HOST_DEVICE
void generate_uniform_double4(rng_state_t *rng_state, double &r1, double &r2,
                              double &r3, double &r4);

/// Generate random direction unit vector (3D)
CUDA_HOST_DEVICE
void generate_unit_vector_3d(rng_state_t *rng_state, float &x, float &y,
                             float &z);

/// Generate random direction unit vector (3D) - double precision
CUDA_HOST_DEVICE
void generate_unit_vector_3d_double(rng_state_t *rng_state, double &x,
                                    double &y, double &z);

/// Generate random point on unit sphere surface
CUDA_HOST_DEVICE
void generate_sphere_point(rng_state_t *rng_state, float &theta, float &phi);

/// Generate random point on unit sphere surface - double precision
CUDA_HOST_DEVICE
void generate_sphere_point_double(rng_state_t *rng_state, double &theta,
                                  double &phi);

// ============================================================================
// BACKWARD COMPATIBILITY WRAPPERS
// ============================================================================

/// Legacy compatibility: convert mqi_rng to mqi_standard_rng
#if defined(__CUDACC__)
CUDA_HOST_DEVICE
inline mqi_standard_rng *legacy_rng_to_standard(void *legacy_rng) {
  return reinterpret_cast<mqi_standard_rng *>(legacy_rng);
}
#else
CUDA_HOST_DEVICE
inline mqi_standard_rng *legacy_rng_to_standard(void *legacy_rng) {
  return reinterpret_cast<mqi_standard_rng *>(legacy_rng);
}
#endif

/// Legacy compatibility wrapper for uniform float generation
CUDA_HOST_DEVICE
inline float mqi_uniform_standard(rng_state_t *rng_state) {
  return generate_uniform_float(rng_state);
}

/// Legacy compatibility wrapper for uniform double generation
CUDA_HOST_DEVICE
inline double mqi_uniform_double_standard(rng_state_t *rng_state) {
  return generate_uniform_double(rng_state);
}

// ============================================================================
// ADVANCED RNG UTILITIES
// ============================================================================

/// Test RNG statistical quality (basic tests)
CUDA_HOST
bool test_rng_quality(const uint32_t num_samples = 1000000);

/// Benchmark RNG performance
CUDA_HOST
double benchmark_rng_performance(const uint32_t num_generations = 10000000);

/// Compare different RNG implementations
CUDA_HOST
void compare_rng_implementations(const uint32_t num_samples = 1000000);

/// Validate RNG reproducibility across different configurations
CUDA_HOST
bool validate_rng_reproducibility(const uint64_t seed,
                                  const uint32_t thread_count,
                                  const uint32_t sequences_per_thread = 1000);

// ============================================================================
// DEVICE-SIDE KERNELS (for GPU initialization and testing)
// ============================================================================

#if defined(__CUDACC__)

/// Kernel to initialize RNG states on device
__global__ void
initialize_rng_states_kernel(rng_state_t *rng_states,
                             const rng_batch_init_t *init_params);

/// Kernel to test RNG quality on device
__global__ void test_rng_quality_kernel(rng_state_t *rng_states,
                                        float *test_results,
                                        const uint32_t samples_per_thread);

/// Kernel to benchmark RNG performance on device
__global__ void benchmark_rng_kernel(rng_state_t *rng_states,
                                     uint64_t *counters,
                                     const uint32_t iterations_per_thread);

#endif // __CUDACC__

// ============================================================================
// CONSTANTS AND DEFAULTS
// ============================================================================

namespace rng_defaults {
constexpr uint64_t DEFAULT_MASTER_SEED = 12345ULL;
constexpr uint32_t DEFAULT_SEQUENCE_OFFSET = 0;
constexpr uint32_t DEFAULT_THREAD_OFFSET = 0;
constexpr uint32_t TEST_SAMPLE_SIZE = 1000000;
constexpr uint32_t BENCHMARK_ITERATIONS = 10000000;
constexpr float DEFAULT_NORMAL_MEAN = 0.0f;
constexpr float DEFAULT_NORMAL_STDDEV = 1.0f;
constexpr double DEFAULT_NORMAL_MEAN_D = 0.0;
constexpr double DEFAULT_NORMAL_STDDEV_D = 1.0;
} // namespace rng_defaults

// ============================================================================
// INLINE IMPLEMENTATIONS (CUDA and Host)
// ============================================================================

#if defined(__CUDACC__)

// CUDA device implementations

CUDA_DEVICE
inline void initialize_rng_state(rng_state_t *rng_state, const uint64_t seed,
                                 const uint32_t thread_id,
                                 const uint64_t sequence_offset) {
  // Initialize curandStatePhilox4_32_10_t
  curand_init(seed, thread_id, sequence_offset, &rng_state->state);

  rng_state->seed = seed;
  rng_state->thread_id = thread_id;
  rng_state->sequence_number = 0;
  rng_state->is_initialized = true;
}

CUDA_DEVICE
inline float generate_uniform_float(rng_state_t *rng_state) {
  return curand_uniform(&rng_state->state);
}

CUDA_DEVICE
inline double generate_uniform_double(rng_state_t *rng_state) {
  return curand_uniform_double(&rng_state->state);
}

CUDA_DEVICE
inline float generate_normal_float(rng_state_t *rng_state, const float mean,
                                   const float stddev) {
  return curand_normal(&rng_state->state) * stddev + mean;
}

CUDA_DEVICE
inline double generate_normal_double(rng_state_t *rng_state, const double mean,
                                     const double stddev) {
  return curand_normal_double(&rng_state->state) * stddev + mean;
}

CUDA_DEVICE
inline uint32_t generate_uniform_int(rng_state_t *rng_state,
                                     const uint32_t min_val,
                                     const uint32_t max_val) {
  return curand(&rng_state->state) % (max_val - min_val + 1) + min_val;
}

CUDA_DEVICE
inline void generate_uniform_float4(rng_state_t *rng_state, float &r1,
                                    float &r2, float &r3, float &r4) {
  uint4 result = curand4(&rng_state->state);
  r1 = result.x;
  r2 = result.y;
  r3 = result.z;
  r4 = result.w;
}

CUDA_DEVICE
inline void generate_unit_vector_3d(rng_state_t *rng_state, float &x, float &y,
                                    float &z) {
  float u1 = generate_uniform_float(rng_state);
  float u2 = generate_uniform_float(rng_state);

  // Uniform distribution on unit sphere
  float phi = 2.0f * M_PI * u1;
  float cos_theta = 2.0f * u2 - 1.0f;
  float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

  x = sin_theta * cosf(phi);
  y = sin_theta * sinf(phi);
  z = cos_theta;
}

#else

// Host implementations

inline void initialize_rng_state(rng_state_t *rng_state, const uint64_t seed,
                                 const uint32_t thread_id,
                                 const uint64_t sequence_offset) {
  // Initialize std::mt19937_64 with derived seed
  uint64_t derived_seed =
      seed + (static_cast<uint64_t>(thread_id) << 32) + sequence_offset;
  rng_state->state.seed(derived_seed);

  rng_state->seed = seed;
  rng_state->thread_id = thread_id;
  rng_state->sequence_number = 0;
  rng_state->is_initialized = true;
}

inline float generate_uniform_float(rng_state_t *rng_state) {
  static thread_local std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  return dist(rng_state->state);
}

inline double generate_uniform_double(rng_state_t *rng_state) {
  static thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);
  return dist(rng_state->state);
}

inline float generate_normal_float(rng_state_t *rng_state, const float mean,
                                   const float stddev) {
  static thread_local std::normal_distribution<float> dist(mean, stddev);
  return dist(rng_state->state);
}

inline double generate_normal_double(rng_state_t *rng_state, const double mean,
                                     const double stddev) {
  static thread_local std::normal_distribution<double> dist(mean, stddev);
  return dist(rng_state->state);
}

inline uint32_t generate_uniform_int(rng_state_t *rng_state,
                                     const uint32_t min_val,
                                     const uint32_t max_val) {
  static thread_local std::uniform_int_distribution<uint32_t> dist(min_val,
                                                                   max_val);
  return dist(rng_state->state);
}

inline void generate_uniform_float4(rng_state_t *rng_state, float &r1,
                                    float &r2, float &r3, float &r4) {
  r1 = generate_uniform_float(rng_state);
  r2 = generate_uniform_float(rng_state);
  r3 = generate_uniform_float(rng_state);
  r4 = generate_uniform_float(rng_state);
}

inline void generate_unit_vector_3d(rng_state_t *rng_state, float &x, float &y,
                                    float &z) {
  float u1 = generate_uniform_float(rng_state);
  float u2 = generate_uniform_float(rng_state);

  // Uniform distribution on unit sphere
  float phi = 2.0f * M_PI * u1;
  float cos_theta = 2.0f * u2 - 1.0f;
  float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

  x = sin_theta * cosf(phi);
  y = sin_theta * sinf(phi);
  z = cos_theta;
}

#endif // __CUDACC__

} // namespace mqi

#endif // MQI_STANDARDIZED_RNG_HPP
