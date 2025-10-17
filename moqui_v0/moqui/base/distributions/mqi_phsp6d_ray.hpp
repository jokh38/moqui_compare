#ifndef MQI_PHSP6D_RAY_H
#define MQI_PHSP6D_RAY_H

/// \file
///
/// Distribution functions (meta-header file for all distributions)

#include <moqui/base/distributions/mqi_pdfMd.hpp>

namespace mqi
{
/// \class phsp_6d
///
/// 6-dimensional uniform pdf for phase-space variables for a spot
//  phase-space variables are position (x,y,z) and direction (x',y',z')
/// \tparam T type of return value
template<typename T>
class phsp_6d_ray : public pdf_Md<T, 6>
{

    /// correlations for x-x' and y-y' respectively
    std::array<T, 2> rho_;   ///< For X,Y
                             //    float            sid;
    float source_position;

public:
    /// Random engine and distribution function
    //    std::default_random_engine  gen_;
    std::normal_distribution<T> func_;

    /// Constructor to initializes mean, sigma, rho, and random engine
    /// \param m[0,1,2]: mean spot-position  of x, y, z
    /// \param m[3,4,5]: mean spot-direction of x', y', z'.
    /// \param s[0,1,2]: std  spot-position of x,y,z -> spot-size
    /// \param s[3,4,5]: std  spot-direction of x,y,z. s[5] is ignored but calculated internally
    /// \param r[0] : x, xp correlation
    /// \param r[1] : y, yp correlation
    /// \note seed setup needs to be done by public method
    /// gen_.seed(from outside, topas or UI); //Ideally set from TOPAS?
    CUDA_HOST_DEVICE
    phsp_6d_ray(std::array<T, 6>& m,
                std::array<T, 6>& s,
                std::array<T, 2>& r,
                float             source_position) :
        pdf_Md<T, 6>(m, s),
        rho_(r) {
        //#if !defined(__CUDACC__)
        //        gen_.seed(std::chrono::system_clock::now().time_since_epoch().count());
        //        gen_.seed(1000);
        func_                 = std::normal_distribution<T>(0, 1);
        this->source_position = source_position;
        //#endif
    }

    /// Constructor to initializes mean, sigma, rho, and random engine
    CUDA_HOST_DEVICE
    phsp_6d_ray(const std::array<T, 6>& m,
                const std::array<T, 6>& s,
                const std::array<T, 6>& r,
                const float             source_position) :
        pdf_Md<T, 6>(m, s),
        rho_(r) {
        //#if !defined(__CUDACC__)
        //        gen_.seed(std::chrono::system_clock::now().time_since_epoch().count());
        //        gen_.seed(1000);
        func_                 = std::normal_distribution<T>(0, 1);
        this->source_position = source_position;
        //#endif
    }

    /// Sample 6 phase-space variables and returns
    CUDA_HOST_DEVICE
    virtual std::array<T, 6>
    operator()(std::default_random_engine* rng) {
        std::array<T, 6> phsp = pdf_Md<T, 6>::mean_;
        T                Ux   = func_(*rng);
        T                Vx   = func_(*rng);
        T                Uy   = func_(*rng);
        T                Vy   = func_(*rng);
        T                Uz   = func_(*rng);   //T Vz = func_(rng);
        T                z    = this->source_position;
        T                A0_x = rho_[0] * rho_[0];
        T                A1_x = pdf_Md<T, 6>::sigma_[3];
        T                A2_x = pdf_Md<T, 6>::sigma_[0] * pdf_Md<T, 6>::sigma_[0];
        T                A0_y = rho_[1] * rho_[1];
        T                A1_y = pdf_Md<T, 6>::sigma_[4];
        T                A2_y = pdf_Md<T, 6>::sigma_[1] * pdf_Md<T, 6>::sigma_[1];
        A2_x                  = A2_x + 2 * A1_x * z + A0_x * z * z;
        A1_x                  = A1_x + A0_x * z;
        A2_y                  = A2_y + 2 * A1_y * z + A0_y * z * z;
        A1_y                  = A1_y + A0_y * z;
        //        printf("A0 %f A1 %f A2 %f\n", A0, A1, A2);
        phsp[0] += std::sqrt(A2_x) * Ux;
        phsp[1] += std::sqrt(A2_y) * Uy;
        phsp[2] += pdf_Md<T, 6>::sigma_[2] * Uz;
        T th20_x = 2 * A0_x;
        if (A2_x > 0.0) {
            phsp[3] += std::sqrt(A2_x) * Ux * A1_x / A2_x;
            th20_x -= 2.0 * A1_x * A1_x / A2_x;
        }
        T th20_y = 2 * A0_y;
        if (A2_y > 0.0) {
            phsp[4] += std::sqrt(A2_y) * Uy * A1_y / A2_y;
            th20_y -= 2.0 * A1_y * A1_y / A2_y;
        }
        T norm = std::sqrt(phsp[3] * phsp[3] + phsp[4] * phsp[4] + phsp[5] * phsp[5]);
        //        printf(
        //          "dir.x %f dir.y %f dir.z %f norm %f th20 %f\n", phsp[3], phsp[4], phsp[5], norm, th20);
        phsp[3] /= norm;
        phsp[4] /= norm;
        phsp[5] /= norm;
        //        printf("Vx %f cos %f sin %f %f\n",
        //               Vx,
        //               std::cos(Vx * std::sqrt(th20 / 2)),
        //               std::sqrt(1 - phsp[3] * phsp[3]),
        //               std::sin(Vx * std::sqrt(th20 / 2)));
        //        printf("dir1 %f %f %f\n",phsp[3]*phsp[3],phsp[4]*phsp[4],phsp[5]*phsp[5]);
        phsp[3] = phsp[3] * std::cos(Vx * std::sqrt(th20_x / 2)) +
                  std::sqrt(1 - phsp[3] * phsp[3]) * std::sin(Vx * std::sqrt(th20_x / 2));
        phsp[4] = phsp[4] * std::cos(Vy * std::sqrt(th20_y / 2)) +
                  std::sqrt(1 - phsp[4] * phsp[4]) * std::sin(Vy * std::sqrt(th20_y / 2));
        //        printf("dir %f %f\n",phsp[3]*phsp[3],phsp[4]*phsp[4]);
        phsp[5] = -1.0 * std::sqrt(1.0 - phsp[3] * phsp[3] - phsp[4] * phsp[4]);
        return phsp;
    };

    //    //// for Raystation
    //    /// Sample 6 phase-space variables and returns
    //    CUDA_HOST_DEVICE
    //    virtual std::array<T, 6>
    //    operator()(std::default_random_engine* rng) {
    //        std::array<T, 6> phsp = pdf_Md<T, 6>::mean_;
    //        T                Ux   = func_(*rng);
    //        T                Vx   = func_(*rng);
    //        T                Uy   = func_(*rng);
    //        T                Vy   = func_(*rng);
    //        T                Uz   = func_(*rng);   //T Vz = func_(rng);
    //        phsp[0] += pdf_Md<T, 6>::sigma_[0] * Ux;
    //        phsp[1] += pdf_Md<T, 6>::sigma_[1] * Uy;
    //        phsp[2] += pdf_Md<T, 6>::sigma_[2] * Uz;
    //
    //        phsp[3] += (pdf_Md<T, 6>::sigma_[3] * Vx + pdf_Md<T, 6>::sigma_[0] * Ux * rho_[0]);
    //        phsp[4] += (pdf_Md<T, 6>::sigma_[4] * VY + pdf_Md<T, 6>::sigma_[1] * Uy * rho_[1]);
    //        //        phsp[3] +=
    //        //          pdf_Md<T, 6>::sigma_[3] * (rho_[0] * Ux + Vx * std::sqrt(1.0 - rho_[0] * rho_[0]));
    //        //        phsp[4] +=
    //        //          pdf_Md<T, 6>::sigma_[4] * (rho_[1] * Uy + Vy * std::sqrt(1.0 - rho_[1] * rho_[1]));
    //        phsp[5] = -1.0 * std::sqrt(1.0 - phsp[3] * phsp[3] - phsp[4] * phsp[4]);
    //        return phsp;
    //    };
};

}   // namespace mqi
#endif
