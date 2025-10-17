#ifndef MQI_BASE_THREAD_HPP
#define MQI_BASE_THREAD_HPP

#include "mqi_common.hpp"
#include "mqi_math.hpp"
#include "mqi_standardized_rng.hpp"

namespace mqi {

///< thread struct to hold random generator
///< thread local - updated to use standardized RNG for Phase 1.0
struct thrd_t {
  uint32_t histories[2]; // histories from and to
  mqi_standard_rng
      rnd_generator;    // Standardized RNG (curandStatePhilox4_32_10_t)
  uint32_t thread_id;   // Thread identifier for RNG initialization
  bool rng_initialized; // RNG initialization status
};

///< random number initialization before entering the mc loop.
///< Updated for Phase 1.0 to use standardized RNG (curandStatePhilox4_32_10_t)
CUDA_GLOBAL
inline void initialize_threads(mqi::thrd_t *thrds, const uint32_t n_threads,
                               unsigned long master_seed = 0,
                               unsigned long offset = 0) {
#if defined(__CUDACC__)
  uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_id < n_threads) {
    thrds[thread_id].thread_id = thread_id;

    // Initialize standardized RNG with Philox4_32_10_t algorithm
    curand_init(master_seed + blockIdx.x, threadIdx.x, offset,
                &thrds[thread_id].rnd_generator);
    thrds[thread_id].rng_initialized = true;
  }
#else
  for (uint32_t i = 0; i < n_threads; ++i) {
    thrds[i].thread_id = i;

    // Initialize host fallback RNG with derived seed
    uint64_t derived_seed = master_seed + i;
    thrds[i].rnd_generator.seed(derived_seed);
    thrds[i].rng_initialized = true;
  }
#endif
}

///< Backward compatibility wrapper for legacy code
CUDA_GLOBAL
inline void initialize_threads_legacy(mqi::thrd_t *thrds,
                                      const uint32_t n_threads,
                                      unsigned long master_seed = 0,
                                      unsigned long offset = 0) {
  // Legacy initialization using old curandState_t approach
  initialize_threads(thrds, n_threads, master_seed, offset);
}

} // namespace mqi

#endif // MQI_BASE_THREAD_HPP
