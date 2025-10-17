#include "mqi_standardized_rng.hpp"
#include "mqi_error_check.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

namespace mqi {

// ============================================================================
// HOST-SIDE IMPLEMENTATIONS
// ============================================================================

CUDA_HOST
void initialize_rng_batch(rng_state_t *rng_states,
                          const rng_batch_init_t *init_params) {
  if (!rng_states || !init_params) {
    return;
  }

  for (uint32_t i = 0; i < init_params->num_threads; ++i) {
    uint32_t thread_id = init_params->thread_offset + i;
    uint64_t sequence_offset = init_params->sequence_offset + i;

    initialize_rng_state(&rng_states[i], init_params->master_seed, thread_id,
                         sequence_offset);
  }
}

CUDA_HOST_DEVICE
void skip_ahead(rng_state_t *rng_state, const uint64_t steps) {
#if defined(__CUDACC__)
  // CUDA skip ahead for curandStatePhilox4_32_10_t
  curand_skipahead(steps, &rng_state->state);
#else
  // Host implementation: advance the generator
  for (uint64_t i = 0; i < steps; ++i) {
    rng_state->state.discard(1);
  }
#endif

  rng_state->sequence_number += steps;
}

CUDA_HOST_DEVICE
void generate_uniform_double4(rng_state_t *rng_state, double &r1, double &r2,
                              double &r3, double &r4) {
  r1 = generate_uniform_double(rng_state);
  r2 = generate_uniform_double(rng_state);
  r3 = generate_uniform_double(rng_state);
  r4 = generate_uniform_double(rng_state);
}

CUDA_HOST_DEVICE
void generate_unit_vector_3d_double(rng_state_t *rng_state, double &x,
                                    double &y, double &z) {
  double u1 = generate_uniform_double(rng_state);
  double u2 = generate_uniform_double(rng_state);

  // Uniform distribution on unit sphere
  double phi = 2.0 * M_PI * u1;
  double cos_theta = 2.0 * u2 - 1.0;
  double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

  x = sin_theta * cos(phi);
  y = sin_theta * sin(phi);
  z = cos_theta;
}

CUDA_HOST_DEVICE
void generate_sphere_point(rng_state_t *rng_state, float &theta, float &phi) {
  float u1 = generate_uniform_float(rng_state);
  float u2 = generate_uniform_float(rng_state);

  // Spherical coordinates
  theta = acosf(2.0f * u2 - 1.0f); // [0, pi]
  phi = 2.0f * M_PI * u1;          // [0, 2pi]
}

CUDA_HOST_DEVICE
void generate_sphere_point_double(rng_state_t *rng_state, double &theta,
                                  double &phi) {
  double u1 = generate_uniform_double(rng_state);
  double u2 = generate_uniform_double(rng_state);

  // Spherical coordinates
  theta = acos(2.0 * u2 - 1.0); // [0, pi]
  phi = 2.0 * M_PI * u1;        // [0, 2pi]
}

// ============================================================================
// ADVANCED RNG UTILITIES
// ============================================================================

CUDA_HOST
bool test_rng_quality(const uint32_t num_samples) {
  std::cout << "Testing RNG quality with " << num_samples << " samples..."
            << std::endl;

  // Create test RNG state
  rng_state_t test_rng;
  initialize_rng_state(&test_rng, rng_defaults::DEFAULT_MASTER_SEED, 0);

  // Test 1: Uniformity test (Chi-square)
  const uint32_t num_bins = 100;
  std::vector<uint32_t> histogram(num_bins, 0);

  for (uint32_t i = 0; i < num_samples; ++i) {
    float val = generate_uniform_float(&test_rng);
    if (val >= 0.0f && val < 1.0f) {
      uint32_t bin = static_cast<uint32_t>(val * num_bins);
      if (bin >= num_bins)
        bin = num_bins - 1;
      histogram[bin]++;
    }
    // Discard invalid values (shouldn't happen with correct implementation)
  }

  // Calculate chi-square statistic
  double expected = static_cast<double>(num_samples) / num_bins;
  double chi_square = 0.0;

  for (uint32_t bin = 0; bin < num_bins; ++bin) {
    double observed = histogram[bin];
    double deviation = observed - expected;
    chi_square += (deviation * deviation) / expected;
  }

  // Chi-square test with 99 degrees of freedom (100 bins - 1)
  // Critical value at p=0.05 is approximately 124.34
  double chi_square_critical = 124.34;
  bool chi_square_passed = chi_square < chi_square_critical;

  std::cout << "Chi-square test: " << chi_square
            << " (critical: " << chi_square_critical << ") - "
            << (chi_square_passed ? "PASSED" : "FAILED") << std::endl;

  // Test 2: Mean test (should be ~0.5)
  double sum = 0.0;
  initialize_rng_state(&test_rng, rng_defaults::DEFAULT_MASTER_SEED, 0);

  for (uint32_t i = 0; i < num_samples; ++i) {
    sum += generate_uniform_float(&test_rng);
  }

  double mean = sum / num_samples;
  double expected_mean = 0.5;
  double mean_error = fabs(mean - expected_mean);
  bool mean_passed = mean_error < 0.01; // Within 1%

  std::cout << "Mean test: " << mean << " (expected: " << expected_mean
            << ", error: " << mean_error << ") - "
            << (mean_passed ? "PASSED" : "FAILED") << std::endl;

  // Test 3: Serial correlation test
  initialize_rng_state(&test_rng, rng_defaults::DEFAULT_MASTER_SEED, 0);
  double correlation_sum = 0.0;

  float prev_val = generate_uniform_float(&test_rng);
  for (uint32_t i = 1; i < num_samples; ++i) {
    float current_val = generate_uniform_float(&test_rng);
    correlation_sum += (prev_val - 0.5) * (current_val - 0.5);
    prev_val = current_val;
  }

  double correlation = correlation_sum / (num_samples - 1);
  bool correlation_passed = fabs(correlation) < 0.01; // Should be near zero

  std::cout << "Serial correlation test: " << correlation << " - "
            << (correlation_passed ? "PASSED" : "FAILED") << std::endl;

  // Overall result
  bool overall_passed = chi_square_passed && mean_passed && correlation_passed;
  std::cout << "Overall RNG quality test: "
            << (overall_passed ? "PASSED" : "FAILED") << std::endl;

  return overall_passed;
}

CUDA_HOST
double benchmark_rng_performance(const uint32_t num_generations) {
  std::cout << "Benchmarking RNG performance with " << num_generations
            << " generations..." << std::endl;

  rng_state_t bench_rng;
  initialize_rng_state(&bench_rng, rng_defaults::DEFAULT_MASTER_SEED, 0);

  // Warm up
  for (uint32_t i = 0; i < 1000; ++i) {
    generate_uniform_float(&bench_rng);
  }

  // Benchmark
  auto start_time = std::chrono::high_resolution_clock::now();

  volatile float sink = 0.0f; // Prevent optimization
  for (uint32_t i = 0; i < num_generations; ++i) {
    sink += generate_uniform_float(&bench_rng);
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);

  double time_us = static_cast<double>(duration.count());
  double generations_per_second =
      static_cast<double>(num_generations) / (time_us / 1e6);
  double ns_per_generation = time_us * 1000.0 / num_generations;

  std::cout << "Performance results:" << std::endl;
  std::cout << "  Total time: " << time_us << " microseconds" << std::endl;
  std::cout << "  Generations per second: " << std::scientific
            << std::setprecision(3) << generations_per_second << std::endl;
  std::cout << "  Nanoseconds per generation: " << std::fixed
            << std::setprecision(3) << ns_per_generation << std::endl;

  return ns_per_generation;
}

CUDA_HOST
void compare_rng_implementations(const uint32_t num_samples) {
  std::cout << "Comparing RNG implementations with " << num_samples
            << " samples..." << std::endl;

  // Test standardized RNG (curandStatePhilox4_32_10_t equivalent)
  rng_state_t standard_rng;
  initialize_rng_state(&standard_rng, rng_defaults::DEFAULT_MASTER_SEED, 0);

  auto start = std::chrono::high_resolution_clock::now();
  double standard_sum = 0.0;
  for (uint32_t i = 0; i < num_samples; ++i) {
    standard_sum += generate_uniform_double(&standard_rng);
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto standard_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  // Test legacy RNG (std::default_random_engine)
  std::default_random_engine legacy_rng(rng_defaults::DEFAULT_MASTER_SEED);
  std::uniform_real_distribution<double> legacy_dist(0.0, 1.0);

  start = std::chrono::high_resolution_clock::now();
  double legacy_sum = 0.0;
  for (uint32_t i = 0; i < num_samples; ++i) {
    legacy_sum += legacy_dist(legacy_rng);
  }
  end = std::chrono::high_resolution_clock::now();
  auto legacy_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  // Test high-quality RNG (std::mt19937_64)
  std::mt19937_64 high_quality_rng(rng_defaults::DEFAULT_MASTER_SEED);
  std::uniform_real_distribution<double> high_quality_dist(0.0, 1.0);

  start = std::chrono::high_resolution_clock::now();
  double high_quality_sum = 0.0;
  for (uint32_t i = 0; i < num_samples; ++i) {
    high_quality_sum += high_quality_dist(high_quality_rng);
  }
  end = std::chrono::high_resolution_clock::now();
  auto high_quality_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  // Results
  std::cout << "RNG Implementation Comparison:" << std::endl;
  std::cout << "Standard (Philox4_32_10): " << standard_sum / num_samples
            << " (time: " << standard_time.count() << " μs)" << std::endl;
  std::cout << "Legacy (default_random_engine): " << legacy_sum / num_samples
            << " (time: " << legacy_time.count() << " μs)" << std::endl;
  std::cout << "High Quality (mt19937_64): " << high_quality_sum / num_samples
            << " (time: " << high_quality_time.count() << " μs)" << std::endl;

  double standard_speed =
      static_cast<double>(num_samples) / standard_time.count();
  double legacy_speed = static_cast<double>(num_samples) / legacy_time.count();
  double high_quality_speed =
      static_cast<double>(num_samples) / high_quality_time.count();

  std::cout << "Speed (M generations/sec):" << std::endl;
  std::cout << "  Standard: " << std::fixed << std::setprecision(3)
            << standard_speed << std::endl;
  std::cout << "  Legacy: " << std::fixed << std::setprecision(3)
            << legacy_speed << std::endl;
  std::cout << "  High Quality: " << std::fixed << std::setprecision(3)
            << high_quality_speed << std::endl;
}

CUDA_HOST
bool validate_rng_reproducibility(const uint64_t seed,
                                  const uint32_t thread_count,
                                  const uint32_t sequences_per_thread) {
  std::cout << "Validating RNG reproducibility with seed=" << seed
            << ", threads=" << thread_count
            << ", sequences=" << sequences_per_thread << std::endl;

  // First run: generate sequences
  std::vector<std::vector<float>> first_run(thread_count);
  for (uint32_t t = 0; t < thread_count; ++t) {
    first_run[t].resize(sequences_per_thread);
    rng_state_t rng;
    initialize_rng_state(&rng, seed, t);

    for (uint32_t s = 0; s < sequences_per_thread; ++s) {
      first_run[t][s] = generate_uniform_float(&rng);
    }
  }

  // Second run: generate sequences with same seed
  std::vector<std::vector<float>> second_run(thread_count);
  for (uint32_t t = 0; t < thread_count; ++t) {
    second_run[t].resize(sequences_per_thread);
    rng_state_t rng;
    initialize_rng_state(&rng, seed, t);

    for (uint32_t s = 0; s < sequences_per_thread; ++s) {
      second_run[t][s] = generate_uniform_float(&rng);
    }
  }

  // Compare results
  bool reproducibility_passed = true;
  for (uint32_t t = 0; t < thread_count; ++t) {
    for (uint32_t s = 0; s < sequences_per_thread; ++s) {
      if (std::abs(first_run[t][s] - second_run[t][s]) > 1e-6f) {
        reproducibility_passed = false;
        break;
      }
    }
    if (!reproducibility_passed)
      break;
  }

  std::cout << "RNG reproducibility test: "
            << (reproducibility_passed ? "PASSED" : "FAILED") << std::endl;

  if (!reproducibility_passed) {
    std::cout << "First few values from thread 0:" << std::endl;
    std::cout << "  Run 1: ";
    for (uint32_t i = 0; i < std::min(5u, sequences_per_thread); ++i) {
      std::cout << first_run[0][i] << " ";
    }
    std::cout << std::endl;
    std::cout << "  Run 2: ";
    for (uint32_t i = 0; i < std::min(5u, sequences_per_thread); ++i) {
      std::cout << second_run[0][i] << " ";
    }
    std::cout << std::endl;
  }

  return reproducibility_passed;
}

// ============================================================================
// CUDA KERNEL IMPLEMENTATIONS
// ============================================================================

#if defined(__CUDACC__)

__global__ void
initialize_rng_states_kernel(rng_state_t *rng_states,
                             const rng_batch_init_t *init_params) {
  const uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_id < init_params->num_threads) {
    uint32_t global_thread_id = init_params->thread_offset + thread_id;
    uint64_t sequence_offset = init_params->sequence_offset + thread_id;

    initialize_rng_state(&rng_states[thread_id], init_params->master_seed,
                         global_thread_id, sequence_offset);
  }
}

__global__ void test_rng_quality_kernel(rng_state_t *rng_states,
                                        float *test_results,
                                        const uint32_t samples_per_thread) {
  const uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  rng_state_t *rng = &rng_states[thread_id];

  // Simple test: calculate mean and variance
  double sum = 0.0;
  double sum_sq = 0.0;

  for (uint32_t i = 0; i < samples_per_thread; ++i) {
    float val = generate_uniform_float(rng);
    sum += val;
    sum_sq += val * val;
  }

  double mean = sum / samples_per_thread;
  double variance = (sum_sq / samples_per_thread) - (mean * mean);

  // Store results (mean should be ~0.5, variance should be ~1/12 ≈ 0.08333)
  test_results[thread_id * 2] = mean;
  test_results[thread_id * 2 + 1] = variance;
}

__global__ void benchmark_rng_kernel(rng_state_t *rng_states,
                                     uint64_t *counters,
                                     const uint32_t iterations_per_thread) {
  const uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  rng_state_t *rng = &rng_states[thread_id];

  uint64_t start_clock = clock64();

  for (uint32_t i = 0; i < iterations_per_thread; ++i) {
    generate_uniform_float(rng);
  }

  uint64_t end_clock = clock64();
  counters[thread_id] = end_clock - start_clock;
}

#endif // __CUDACC__

} // namespace mqi
