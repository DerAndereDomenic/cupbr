#ifndef __CUPBR_GEOMETRY_MATERIALDETAIL_CUH
#define __CUPBR_GEOMETRY_MATERIALDETAIL_CUH

__host__ __device__
Vector3float
Material::brdf(const Vector3float& position, const Vector3float& inc_dir, const Vector3float& out_dir)
{
    return Vector3float(0);
}

__host__ __device__
Vector3float
Material::brdf_lambert(const Vector3float& position, const Vector3float& inc_dir, const Vector3float& out_dir)
{
    return Vector3float(0);
}

__host__ __device__
Vector3float
Material::brdf_phong(const Vector3float& position, const Vector3float& inc_dir, const Vector3float& out_dir)
{
    return Vector3float(0);
}

#endif