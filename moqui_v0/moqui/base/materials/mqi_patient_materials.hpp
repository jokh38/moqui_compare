#ifndef MQI_PATIENT_MATERIAL_HPP
#define MQI_PATIENT_MATERIAL_HPP

#include <cassert>
#include <moqui/base/materials/mqi_material.hpp>
#include <moqui/base/mqi_common.hpp>
#include <moqui/base/mqi_physics_constants.hpp>

namespace mqi
{

///< Interaction model (pure virtual class)
///< interface between particle and material
///< template R and particle type
///<
template<typename R>
class patient_material_t : public h2o_t<R>
{

public:
    CUDA_HOST_DEVICE
    patient_material_t() : h2o_t<R>() {
        ;
    }

    CUDA_HOST_DEVICE
    patient_material_t(int16_t hu) : h2o_t<R>() {
        this->rho_mass = hu_to_density(hu);
        this->X0       = radiation_length(this->rho_mass);
    }

    CUDA_HOST_DEVICE
    ~patient_material_t() {
        ;
    }
    /// <calculate density based on HU value
    /// param : HU (unitless)
    /// return: density (g/mm^-3)
    /// Based on Schneider conversion in TOPAS
    /// W. Schneider, et al., Phys. Med. Biol., 1996 ?? Which Schneider, not sure.
    CUDA_HOST_DEVICE
    virtual R
    hu_to_density(int16_t hu) {
        if (hu < -1000) 
        {
            hu = -1000;
        } 
        else if (hu > 6000) 
        {
            hu = 6000;
        }
        R rho_mass = 0.0;

        // HU to density curve SMC added by Chanil Jeon
        // Piecewise-linear curve

        if (hu >= -1000 && hu < -723) rho_mass = 0.97509 + 0.00097388 * hu;
        else if (hu >= -723 && hu < -553) rho_mass = 0.95141 + 0.00094118 * hu;
        else if (hu >= -553 && hu < -89) rho_mass = 1.0425 + 0.0011056 * hu;
        else if (hu >= -89 && hu < -38) rho_mass = 1.0171 + 0.00082353 * hu;
        else if (hu >= -38 && hu < 4) rho_mass = 0.99893 + 0.00035714 * hu;
        else if (hu >= 4 && hu < 6) rho_mass = 0.9745 + 0.0085 * hu;
        else if (hu >= 6 && hu < 29) rho_mass = 1.0092 + 0.0015652 * hu;
        else if (hu >= 29 && hu < 81) rho_mass = 1.0293 + 0.00084615 * hu;
        else if (hu >= 81 && hu < 229) rho_mass = 1.0711 + 0.00032432 * hu;
        else if (hu >= 229 && hu < 244) rho_mass = 0.9322 + 0.00093334 * hu;
        else if (hu >= 244 && hu < 461) rho_mass = 0.96191 + 0.00081106 * hu;
        else if (hu >= 461 && hu < 829) rho_mass = 1.0538 + 0.00061141 * hu;
        else if (hu >= 829 && hu < 1248) rho_mass = 1.0363 + 0.00063246 * hu;
        else if (hu >= 1248 && hu <= 6000) rho_mass = 1.1206 + 0.00056491 * hu;
        else if (hu > 6000) rho_mass = 4.60;

        // R correction_factor = density_correction[hu + 1000];
        // rho_mass *= correction_factor;   /// g/cm3

        rho_mass /= 1000.0;   ///g/mm3
        return rho_mass;
    }
};

}   // namespace mqi

#endif
