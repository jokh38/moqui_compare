#include "mqi_math.hpp"

namespace mqi {

#if defined(__CUDACC__)

///< specialization of template functions
// Natural log
template <> float mqi_ln(float s) { return logf(s); }

template <> double mqi_ln(double s) { return log(s); }

///< sqrt
template <> float mqi_sqrt(float s) { return sqrtf(s); }

template <> double mqi_sqrt(double s) { return sqrt(s); }

///< power
template <> float mqi_pow(float s, float p) { return powf(s, p); }

template <> double mqi_pow(double s, double p) { return pow(s, p); }

///< exponential
template <> float mqi_exp(float s) { return expf(s); }

template <> double mqi_exp(double s) { return exp(s); }

///< acos
template <> float mqi_acos(float s) { return acosf(s); }

template <> double mqi_acos(double s) { return acos(s); }

///< cos
template <> float mqi_cos(float s) { return cosf(s); }

template <> double mqi_cos(double s) { return cos(s); }

///< sin
template <> float mqi_sin(float s) { return sinf(s); }

template <> double mqi_sin(double s) { return sin(s); }

template <> float mqi_abs(float s) { return abs(s); }

template <> double mqi_abs(double s) { return abs(s); }

template <> float mqi_round(float s) { return roundf(s); }

template <> double mqi_round(double s) { return round(s); }

template <> float mqi_floor(float s) { return floorf(s); }

template <> double mqi_floor(double s) { return floor(s); }

template <> float mqi_ceil(float s) { return ceilf(s); }

template <> double mqi_ceil(double s) { return ceil(s); }

template <> bool mqi_isnan(float s) {
  bool t = isnan(s);
  return t;
}

template <> bool mqi_isnan(double s) {
  bool t = isnan(s);
  return t;
}

// random number status per thread. each status is initialized by master seed.
// the parameters to be passed are currand_status for CUDA and random_engine for
// C++ curand_status == mqi_rng ;

template <> float mqi_uniform<float>(mqi_rng *rng) {
  return curand_uniform(rng);
}

template <> // template<class S = curandState>
double mqi_uniform<double>(mqi_rng *rng) {
  return curand_uniform_double(rng);
}

template <> float mqi_normal<float>(mqi_rng *rng, float avg, float sig) {
  return curand_normal(rng) * sig + avg;
}

template <> double mqi_normal<double>(mqi_rng *rng, double avg, double sig) {
  return curand_normal_double(rng) * sig + avg;
}

template <> float mqi_exponential<float>(mqi_rng *rng, float avg, float up) {
  float x;
  do {
    x = -1.0 / avg * logf(1.0 - curand_uniform(rng)); // 0, up
  } while (x > up || mqi::mqi_isnan(x));

  return x;
}

template <>
double mqi_exponential<double>(mqi_rng *rng, double avg, double up) {
  double x;
  do {
    x = -1.0 / avg * log(1.0 - curand_uniform(rng)); // 0, up
  } while (x > up || mqi::mqi_isnan(x));
  return x;
}

#else

// Natural log. C++ casts float to double. they have same implementation.
template <> float mqi_ln(float s) { return std::log(s); }

template <> double mqi_ln(double s) { return std::log(s); }

template <> float mqi_sqrt(float s) { return std::sqrt(s); }

template <> double mqi_sqrt(double s) { return std::sqrt(s); }

template <> float mqi_pow(float s, float p) { return std::pow(s, p); }

template <> double mqi_pow(double s, double p) { return std::pow(s, p); }

///< exponential
template <> float mqi_exp(float s) { return std::exp(s); }

template <> double mqi_exp(double s) { return std::exp(s); }

///< acos
template <> float mqi_acos(float s) { return std::acos(s); }

template <> double mqi_acos(double s) { return std::acos(s); }

template <> float mqi_cos(float s) { return std::cos(s); }

template <> double mqi_cos(double s) { return std::cos(s); }

template <> float mqi_abs(float s) { return std::abs(s); }

template <> double mqi_abs(double s) { return std::abs(s); }

///< round
template <> float mqi_round(float s) { return std::roundf(s); }

template <> double mqi_round(double s) { return std::round(s); }

////< floor

template <> float mqi_floor(float s) { return std::floor(s); }

template <> double mqi_floor(double s) { return std::floor(s); }

////< ceil
template <> float mqi_ceil(float s) { return std::ceil(s); }

template <> double mqi_ceil(double s) { return std::ceil(s); }

////< isnan
template <> bool mqi_isnan(float s) { return std::isnan(s); }

template <> bool mqi_isnan(double s) { return std::isnan(s); }

template <> float mqi_uniform<float>(mqi_rng *rng) {
  std::uniform_real_distribution<float> dist;
  return dist(*rng);
}

template <> double mqi_uniform<double>(mqi_rng *rng) {
  std::uniform_real_distribution<double> dist;
  return dist(*rng);
}

template <> float mqi_normal<float>(mqi_rng *rng, float avg, float sig) {
  std::normal_distribution<float> dist(avg, sig);
  return dist(*rng);
}

template <> double mqi_normal<double>(mqi_rng *rng, double avg, double sig) {
  std::normal_distribution<double> dist(avg, sig);
  return dist(*rng);
}

template <> float mqi_exponential<float>(mqi_rng *rng, float avg, float up) {
  float x;
  std::exponential_distribution<float> dist(avg);
  x = dist(*rng);
  //    do {
  //        x = dist(*rng);
  //    } while (x > up || x <= 0);
  (void)up; // Suppress unused parameter warning
  return x;
}

template <>
double mqi_exponential<double>(mqi_rng *rng, double avg, double up) {
  double x;
  std::exponential_distribution<double> dist(avg);
  do {
    x = dist(*rng);
  } while (x > up || x <= 0);
  return x;
}

#endif

// Note: Functions are defined as template specializations above
// No explicit template instantiations needed for template specializations

} // namespace mqi
