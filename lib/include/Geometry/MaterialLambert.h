#ifndef __CUPBR_GEOMETRY_MATERIALLAMBERT_H
#define __CUPBR_GEOMETRY_MATERIALLAMBERT_H

#include <Geometry/Material.h>

namespace cupbr
{
    /**
    *   @brief A class to model different materials
    */
    class MaterialLambert : public Material
    {
        public:
        /**
        *   @brief Default constructor
        */
        MaterialLambert(const Properties& properties)
        {
            type = MaterialType::LAMBERT;
            albedo_e = properties.getProperty("albedo_e", Vector3float(0));
            albedo_d = properties.getProperty("albedo_d", Vector3float(1));
        }

        Vector3float albedo_e = Vector3float(0);                    /**< The emissive albedo */
        Vector3float albedo_d = Vector3float(1);                    /**< The diffuse albedo */

        /**
        *   @brief Compute the brdf of the material
        *   @param[in] position The position
        *   @param[in] inc_dir The incoming direction
        *   @param[in] out_dir The outgoing directio
        *   @return The brdf
        *   @pre dot(inc_dir,out_dir) >= 0
        */
        __host__ __device__
        virtual Vector3float brdf(const Vector3float& position, const Vector3float& inc_dir, const Vector3float& out_dir, const Vector3float& normal);

        /**
        *   @brief Importance sample the material
        *   @param[in] seed The seed for the rng
        *   @param[in] inc_dir The incoming direction
        *   @param[in] normal The surface normal
        *   @return A 4D Vector. The first three components mark the new direction and the w component the sampling probability
        *   @note For glass the w component is 0 for reflection and 1 for refraction
        */
        __host__ __device__
        virtual Vector4float sampleDirection(uint32_t& seed, const Vector3float& inc_dir, const Vector3float& normal);
    };

} //namespace cupbr

#include "../../src/Geometry/MaterialLambertDetail.h"

#endif