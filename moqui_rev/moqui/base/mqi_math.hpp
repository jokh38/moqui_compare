#ifndef MQI_MATH_HPP
#define MQI_MATH_HPP

/// \file
///
/// A header including CUDA related headers and functions

#include "mqi_common.hpp"
#include "mqi_physics_constants.hpp"

#include <cmath>
#include <mutex>
#include <random>

namespace mqi {

// const float near_zero = 0.0000001;
const float near_zero = mqi::physics::NEAR_ZERO<float>;
const float min_step = mqi::physics::MIN_STEP_SIZE<float>;
const float geometry_tolerance = mqi::physics::GEOMETRY_TOLERANCE<float>;

///< TODO: Implement CUDA-specific infinity constants
const float m_inf = -1.0 * HUGE_VALF;
const float p_inf = HUGE_VALF;

template <typename T>
CUDA_DEVICE inline T intpl1d(T x, T x0, T x1, T y0, T y1) {
  return (x1 == x0) ? y0 : y0 + (x - x0) * (y1 - y0) / (x1 - x0);
}

template <typename T> CUDA_DEVICE T mqi_ln(T s);

template <typename T> CUDA_HOST_DEVICE T mqi_sqrt(T s);

template <typename T> CUDA_DEVICE T mqi_pow(T s, T p);

template <typename T> CUDA_DEVICE T mqi_exp(T s);

template <typename T> CUDA_DEVICE T mqi_acos(T s);

template <typename T> CUDA_DEVICE T mqi_cos(T s);

template <typename T> CUDA_DEVICE T mqi_sin(T s);

template <typename T> CUDA_DEVICE T mqi_abs(T s);

template <typename T> CUDA_HOST_DEVICE T mqi_round(T s);

template <typename T> CUDA_HOST_DEVICE T mqi_floor(T s);

template <typename T> CUDA_HOST_DEVICE T mqi_ceil(T s);

template <typename T> CUDA_HOST_DEVICE bool mqi_isnan(T s);

/*
template<typename T>
CUDA_DEVICE
T mqi_inf();
*/

///< To make template for both return type and argument.
///< 1. return type template
template <class T> struct rnd_return {
  typedef T type;
};

template <> struct rnd_return<float> {
  typedef float type;
};

template <> struct rnd_return<double> {
  typedef double type;
};

///< 2. distribution funtion template.

///< normal
template <class T, class S>
CUDA_DEVICE typename rnd_return<T>::type mqi_normal(S *rng, T avg, T sig) {
  return T();
}

///< uniform
template <class T, class S>
CUDA_DEVICE typename rnd_return<T>::type mqi_uniform(S *rng) {
  return T();
}

///< exponetial distribution
template <class T, class S>
CUDA_DEVICE typename rnd_return<T>::type mqi_exponential(S *rng, T avg, T up) {
  return T();
}

// Type definitions for random number generators
#if defined(__CUDACC__)
typedef curandState_t mqi_rng;
#else
typedef std::default_random_engine mqi_rng;
#endif

// Note: Template specializations are defined in mqi_math.cpp
// No explicit template instantiation declarations needed

} // namespace mqi

#endif
