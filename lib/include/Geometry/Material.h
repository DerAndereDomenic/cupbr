#ifndef __CUPBR_GEOMETRY_MATERIAL_H
#define __CUPBR_GEOMETRY_MATERIAL_H

#include <Core/CUDA.h>
#include <Math/Vector.h>
#include <DataStructure/Volume.h>
#include <Core/Properties.h>
#include <Core/Plugin.h>

namespace cupbr
{

    /**
    *   @brief The types of materials supported
    */
    enum class MaterialType
    {
        LAMBERT,
        PHONG,
        MIRROR,
        GLASS,
        GGX,
        VOLUME
    };

    /**
    *   @brief A class to model different materials
    */
    class Material : public Plugin
    {
        public:

        Material() = default;

        /**
        *   @brief Default constructor
        *   @param properties The input properties
        */
        Material(const Properties& properties) {}

        MaterialType type = MaterialType::LAMBERT;                  /**< The material type */
        Volume volume;

        /**
        *   @brief Compute the brdf of the material
        *   @param[in] position The position
        *   @param[in] inc_dir The incoming direction
        *   @param[in] out_dir The outgoing directio
        *   @return The brdf
        *   @pre dot(inc_dir,out_dir) >= 0
        */
        __host__ __device__
        virtual Vector3float brdf(const Vector3float& position, const Vector3float& inc_dir, const Vector3float& out_dir, const Vector3float& normal) { return 0; }

        /**
        *   @brief Importance sample the material
        *   @param[in] seed The seed for the rng
        *   @param[in] inc_dir The incoming direction
        *   @param[in] normal The surface normal
        *   @return A 4D Vector. The first three components mark the new direction and the w component the sampling probability
        *   @note For glass the w component is 0 for reflection and 1 for refraction
        */
        __host__ __device__
        virtual Vector4float sampleDirection(uint32_t& seed, const Vector3float& inc_dir, const Vector3float& normal) { return 0; }

        /**
        *   @brief Henyey greenstein phase function
        *   @param[in] g The scattering parameter
        *   @param[in] cos_theta The angle between incoming and outgoing direction
        *   @return The scattering probability
        */
        __host__ __device__
        static float henyeyGreensteinPhaseFunction(const float& g, const float& cos_theta);

        /**
        *   @brief Sample the henyey greenstein phase function  
        *   @param[in] g The scattering parameter
        *   @param[in] forward The forward scattering direction 
        *   @param[in] seed The seed
        *   @return A 4D vector with the direction and pdf as 4th component
        */
        __host__ __device__
        static Vector4float sampleHenyeyGreensteinPhaseFunction(const float& g, const Vector3float& forward, uint32_t& seed);
    };

} //namespace cupbr

#include "../../src/Geometry/MaterialDetail.h"

#endif