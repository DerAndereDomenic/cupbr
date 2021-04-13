#ifndef __CUPBR_GEOMETRY_MATERIALDETAIL_CUH
#define __CUPBR_GEOMETRY_MATERIALDETAIL_CUH

#include <cmath>
#include <Math/Functions.cuh>

__host__ __device__
inline Vector3float
Material::brdf(const Vector3float& position, const Vector3float& inc_dir, const Vector3float& out_dir, const Vector3float& normal)
{
    switch(type)
    {
        case LAMBERT:
        {
            return brdf_lambert();
        }
        break;
        case PHONG:
        {   
            return brdf_phong(position, inc_dir, out_dir, normal) + brdf_lambert();
        }
        break;
        case MIRROR:
        {
            return brdf_mirror(position, inc_dir, out_dir, normal);
        }
        break;
        case GLASS:
        {
            return btdf_glass(position, inc_dir, out_dir, normal);
        }
        break;
    }

    return Vector3float(0);
}

__host__ __device__
inline Vector3float
Material::brdf_lambert()
{
    return albedo_d/static_cast<float>(M_PI);
}

__host__ __device__
inline Vector3float
Material::brdf_phong(const Vector3float& position, const Vector3float& inc_dir, const Vector3float& out_dir, const Vector3float& normal)
{
    Vector3float halfDir = Math::normalize(inc_dir + out_dir);
    return albedo_s*powf(fmaxf(0.0f,Math::dot(halfDir,normal)), shininess);
}

__host__ __device__
inline Vector3float
Material::brdf_mirror(const Vector3float& position, const Vector3float& inc_dir, const Vector3float& out_dir, const Vector3float& normal)
{
    Vector3float reflected = Math::reflect(inc_dir,normal);
    return albedo_s*Math::delta(1.0f-Math::dot(out_dir,reflected))/Math::dot(out_dir,normal);
}

__host__ __device__
inline Vector3float
Material::btdf_glass(const Vector3float& position, const Vector3float& inc_dir, const Vector3float& out_dir, const Vector3float& normal)
{
    Vector3float refracted = Math::refract(eta, inc_dir, normal);
    return albedo_s*Math::delta(1.0f-Math::dot(refracted,out_dir))/Math::dot(inc_dir,normal);
}


#endif