#ifndef __CUPBR_GEOMETRY_MATERIALDETAIL_CUH
#define __CUPBR_GEOMETRY_MATERIALDETAIL_CUH

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
            printf("Not supported!\n");
        }
        break;
        case GLASS:
        {
            printf("Not supported!\n");
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
    return albedo_s*powf(max(0.0f,Math::dot(halfDir,normal)), shininess);
}

#endif