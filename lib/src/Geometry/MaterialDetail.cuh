#ifndef __CUPBR_GEOMETRY_MATERIALDETAIL_CUH
#define __CUPBR_GEOMETRY_MATERIALDETAIL_CUH

__host__ __device__
Vector3float
Material::brdf()
{
    return Vector3float(0);
}

__host__ __device__
Vector3float
Material::brdf_lambert()
{
    return Vector3float(0);
}

__host__ __device__
Vector3float
Material::brdf_phong()
{
    return Vector3float(0);
}

#endif