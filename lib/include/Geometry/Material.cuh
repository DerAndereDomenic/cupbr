#ifndef __CUPBR_GEOMETRY_MATERIAL_CUH
#define __CUPBR_GEOMETRY_MATERIAL_CUH

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
        brdf(const Vector3float& position, const Vector3float& inc_dir, const Vector3float& out_dir);
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
        brdf_lambert(const Vector3float& position, const Vector3float& inc_dir, const Vector3float& out_dir);

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
        brdf_phong(const Vector3float& position, const Vector3float& inc_dir, const Vector3float& out_dir);
};

#include "../../src/Geometry/MaterialDetail.cuh"

#endif