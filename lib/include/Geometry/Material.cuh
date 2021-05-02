#ifndef __CUPBR_GEOMETRY_MATERIAL_CUH
#define __CUPBR_GEOMETRY_MATERIAL_CUH

#include <Core/CUDA.cuh>
#include <Math/Vector.h>

/**
*   @brief The types of materials supported
*/
enum MaterialType
{
    LAMBERT,
    PHONG,
    MIRROR,
    GLASS
};

/**
*   @brief A class to model different materials
*/ 
class Material
{
    public:
        /**
        *   @brief Default constructor 
        */
        Material() = default;

        Vector3float albedo_d = Vector3float(1);    /**< The diffuse albedo */
        Vector3float albedo_s = Vector3float(0);    /**< The specular albedo */
        float shininess = 128.0f*0.4f;              /**< The object shininess */
        float eta = 1.5f;                           /**< Index of refraction */

        MaterialType type = LAMBERT;                /**< The material type */

        /**
        *   @brief Compute the brdf of the material
        *   @param[in] position The position
        *   @param[in] inc_dir The incoming direction
        *   @param[in] out_dir The outgoing directio
        *   @return The brdf
        *   @pre dot(inc_dir,out_dir) >= 0
        */ 
        __host__ __device__
        Vector3float
        brdf(const Vector3float& position, const Vector3float& inc_dir, const Vector3float& out_dir, const Vector3float& normal);

        /**
        *   @brief Importance sample the material
        *   @param[in] seed The seed for the rng
        *   @param[in] inc_dir The incoming direction
        *   @param[in] normal The surface normal
        *   @return A 4D Vector. The first three components mark the new direction and the w component the sampling probability 
        *   @note For glass the w component is 0 for reflection and 1 for refraction
        */
        __host__ __device__
        Vector4float
        sampleDirection(uint32_t& seed, const Vector3float& inc_dir, const Vector3float& normal);
    private:
        /**
        *   @brief Compute the lambert brdf
        *   @param[in] position The position
        *   @param[in] inc_dir The incoming direction
        *   @param[in] out_dir The outgoing directio
        *   @return The brdf
        *   @pre dot(inc_dir,out_dir) >= 0
        */ 
        __host__ __device__
        Vector3float
        brdf_lambert();

        /**
        *   @brief Compute the phong brdf
        *   @param[in] position The position
        *   @param[in] inc_dir The incoming direction
        *   @param[in] out_dir The outgoing directio
        *   @return The brdf
        *   @pre dot(inc_dir,out_dir) >= 0
        */ 
        __host__ __device__
        Vector3float
        brdf_phong(const Vector3float& position, const Vector3float& inc_dir, const Vector3float& out_dir, const Vector3float& normal);

        /**
        *   @brief Compute the mirror brdf
        *   @param[in] position The position
        *   @param[in] inc_dir The incoming direction
        *   @param[in] out_dir The outgoing directio
        *   @return The brdf
        *   @pre dot(inc_dir,out_dir) >= 0
        */ 
        __host__ __device__
        Vector3float
        brdf_mirror(const Vector3float& position, const Vector3float& inc_dir, const Vector3float& out_dir, const Vector3float& normal);

        /**
        *   @brief Compute glass btdf 
        *   @param[in] position The position
        *   @param[in] inc_dir The incoming direction
        *   @param[in] out_dir The outgoing directio
        *   @return The btdf
        */
       __host__ __device__
       Vector3float
       btdf_glass(const Vector3float& position, const Vector3float& inc_dir, const Vector3float& out_dir, const Vector3float& normal);

        /**
        *   @brief Importance sample lambert material
        *   @param[in] seed The seed for the rng
        *   @param[in] normal The surface normal
        *   @return A 4D Vector. The first three components mark the new direction and the w component the sampling probability 
        */
       __host__ __device__
       Vector4float
       sample_lambert(uint32_t& seed, const Vector3float& normal);

        /**
        *   @brief Importance sample mirror material
        *   @param[in] inc_dir The incoming direction
        *   @param[in] normal The surface normal
        *   @return A 4D Vector. The first three components mark the new direction and the w component the sampling probability 
        */
       __host__ __device__
       Vector4float
       sample_mirror(const Vector3float& inc_dir, const Vector3float& normal);

        /**
        *   @brief Importance sample glass material
        *   @param[in] seed The seed for the rng
        *   @param[in] inc_dir The incoming direction
        *   @param[in] normal The surface normal
        *   @return A 4D Vector. The first three components mark the new direction and the w component the sampling probability 
        *   @note The w component is 0 for reflection and 1 for refraction
        */
       __host__ __device__
       Vector4float
       sample_glass(uint32_t& seed, const Vector3float& inc_dir, const Vector3float& normal);
};

#include "../../src/Geometry/MaterialDetail.cuh"

#endif