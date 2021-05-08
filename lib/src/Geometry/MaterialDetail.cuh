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
inline Vector4float
Material::sampleDirection(uint32_t& seed, const Vector3float& inc_dir, const Vector3float& normal)
{
    switch(type)
    {
        case LAMBERT:
        case PHONG:
        {
            return sample_lambert(seed, normal);
        }
        break;
        case MIRROR:
        {
            return sample_mirror(inc_dir, normal);
        }
        break;
        case GLASS:
        {
            return sample_glass(seed, inc_dir, normal);
        }
        break;
    }

    return Vector4float(INFINITY);
}

__host__ __device__
inline Vector4float
Material::sample_lambert(uint32_t& seed, const Vector3float& normal)
{
    const float xi_1 = Math::rnd(seed);
    const float xi_2 = Math::rnd(seed);

    const float r = sqrtf(xi_1);
    const float phi = 2.0f*3.14159f*xi_2;

    const float x = r*cos(phi);
    const float y = r*sin(phi);
    const float z = sqrtf(fmaxf(0.0f, 1.0f - x*x-y*y));

    Vector3float direction = Math::normalize(Math::toLocalFrame(normal, Vector3float(x,y,z)));

    float p = fmaxf(EPSILON, Math::dot(direction, normal))/3.14159f;

    return Vector4float(direction, p);
}

__host__ __device__
inline Vector4float
Material::sample_mirror(const Vector3float& inc_dir, const Vector3float& normal)
{
    Vector3float direction = Math::reflect(inc_dir, normal);

    return Vector4float(direction, 1.0f);
}

__host__ __device__
inline Vector4float
Material::sample_glass(uint32_t& seed, const Vector3float& inc_dir, const Vector3float& n)
{
    const float NdotV = Math::dot(inc_dir, n);
    bool outside = NdotV > 0.0f;
    float _eta = outside ? 1.0f/eta : eta;
    Vector3float normal = outside ? n : -1.0f*n;
    float F0 = outside ? (1.0f - eta) / (1.0f + eta) : (-1.0f + eta) / (1.0f + eta);
    F0 *= F0;

    float p_reflect = Math::fresnel_schlick(F0, Math::dot(inc_dir, normal));
    float xi = Math::rnd(seed);

    Vector3float refraction_dir = Math::refract(_eta, inc_dir, normal);
    Vector3float direction;
    if(xi <= p_reflect || Math::safeFloatEqual(Math::norm(refraction_dir), 0.0f))
    {
        direction = Math::reflect(inc_dir, normal);
    }
    else
    {
        direction = Math::normalize(refraction_dir);
    }

    return Vector4float(direction, 1);
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
    return albedo_s*Math::delta(1.0f-Math::dot(out_dir,reflected))/Math::dot(inc_dir,normal);
}

__host__ __device__
inline Vector3float
Material::btdf_glass(const Vector3float& position, const Vector3float& inc_dir, const Vector3float& out_dir, const Vector3float& normal)
{
    //TODO
    //Vector3float refracted = Math::refract(eta, inc_dir, normal);
    //return 1;//albedo_s*Math::delta(1.0f-Math::dot(refracted,out_dir))/Math::dot(inc_dir,normal);
    if(Math::dot(inc_dir, out_dir) > 0) //Reflected
    {
        if(Math::dot(inc_dir,normal) > 0)
        {
            return brdf_mirror(position, inc_dir, out_dir, normal);
        }
        else
        {
            return brdf_mirror(position, inc_dir, out_dir, -1.0f*normal);
        }
        
    }
    else{
        Vector3float refracted;
        Vector3float n = normal;
        if(Math::dot(inc_dir,normal) > 0)
        {
            refracted = Math::refract(1.0f/eta, inc_dir, normal);
        }
        else
        {
            refracted = Math::refract(eta, inc_dir, -1.0f*normal);
            n = -1.0f*normal;
        }
        return albedo_s * Math::delta(1.0f-Math::dot(refracted,out_dir))/Math::dot(inc_dir,n);
    }
}


#endif