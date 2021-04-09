#ifndef __CUPBR_GEOMETRY_MATERIAL_CUH
#define __CUPBR_GEOMETRY_MATERIAL_CUH

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

class Material
{
    public:
        Material() = default;

        Vector3float albedo_d = Vector3float(1);
        Vector3float albedo_s = Vector3float(0);

        MaterialType type = LAMBERT;

        __host__ __device__
        Vector3float
        brdf();
    private:
        __host__ __device__
        Vector3float
        brdf_lambert();

        __host__ __device__
        Vector3float
        brdf_phong();
};

#include "../../src/Geometry/MaterialDetail.cuh"

#endif