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

       __host__ __device__
       Vector4float
       sample_lambert(uint32_t& seed, const Vector3float& normal);

       __host__ __device__
       Vector4float
       sample_mirror(const Vector3float& inc_dir, const Vector3float& normal);

       __host__ __device__
       Vector4float
       sample_glass();
};

#include "../../src/Geometry/MaterialDetail.cuh"

#endif