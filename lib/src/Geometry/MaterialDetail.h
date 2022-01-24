#ifndef __CUPBR_GEOMETRY_MATERIALDETAIL_H
#define __CUPBR_GEOMETRY_MATERIALDETAIL_H

#include <cmath>
#include <Math/Functions.h>

namespace cupbr
{
    namespace detail
    {
        __host__ __device__
        inline float
        D_GGX(const float& NdotH, const float roughness)
        {
            float a2 = roughness * roughness;
            float d = (NdotH * a2 - NdotH) * NdotH + 1.0f;
            return a2 / (static_cast<float>(M_PI) * d * d);
        }

        __host__ __device__
        inline float
        V_SmithJointGGX(const float& NdotL, const float& NdotV, const float& roughness)
        {
            float a2 = roughness * roughness;
            float denomA = NdotV * sqrt(a2 + (1.0f - a2) * NdotL * NdotL);
            float denomB = NdotL * sqrt(a2 + (1.0f - a2) * NdotV * NdotV);

            float G = 2.0f * NdotL * NdotV / (denomA + denomB + EPSILON);

            return G / (4.0f * NdotV * NdotL + EPSILON);
        }

    } //namespace detail

    __host__ __device__
    inline Vector3float
    Material::brdf(const Vector3float& position, const Vector3float& inc_dir, const Vector3float& out_dir, const Vector3float& normal)
    {
        switch (type)
        {
            case MaterialType::LAMBERT:
            {
                return brdf_lambert();
            }
            break;
            case MaterialType::PHONG:
            {
                return brdf_phong(position, inc_dir, out_dir, normal) + brdf_lambert();
            }
            break;
            case MaterialType::MIRROR:
            {
                return brdf_mirror(position, inc_dir, out_dir, normal);
            }
            break;
            case MaterialType::GLASS:
            {
                return btdf_glass(position, inc_dir, out_dir, normal);
            }
            break;
            case MaterialType::GGX:
            {
                return brdf_ggx(position, inc_dir, out_dir, normal);
            }
            break;
            case MaterialType::VOLUME:
            {
                return 0;
            }
            break;
        }

        return Vector3float(0);
    }

    __host__ __device__
    inline Vector4float
    Material::sampleDirection(uint32_t& seed, const Vector3float& inc_dir, const Vector3float& normal)
    {
        switch (type)
        {
            case MaterialType::LAMBERT:
            case MaterialType::PHONG:
            {
                return sample_lambert(seed, normal);
            }
            break;
            case MaterialType::MIRROR:
            {
                return sample_mirror(inc_dir, normal);
            }
            break;
            case MaterialType::GLASS:
            {
                return sample_glass(seed, inc_dir, normal);
            }
            break;
            case MaterialType::GGX:
            {
                return sample_ggx(seed, inc_dir, normal);
            }
            break;
            case MaterialType::VOLUME:
            {
                return volume.interface == Interface::GLASS ? sample_glass(seed, inc_dir, normal) : Vector4float(-1.0f * inc_dir, 1);
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
        const float phi = 2.0f * 3.14159f * xi_2;

        const float x = r * cos(phi);
        const float y = r * sin(phi);
        const float z = sqrtf(fmaxf(0.0f, 1.0f - x * x - y * y));

        Vector3float direction = Math::normalize(Math::toLocalFrame(normal, Vector3float(x, y, z)));

        float p = fmaxf(EPSILON, Math::dot(direction, normal)) / 3.14159f;

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
        float _eta = outside ? 1.0f / eta : eta;
        Vector3float normal = outside ? n : -1.0f * n;
        float F0 = (eta - 1.0f) / (eta + 1.0f);
        F0 *= F0;

        float p_reflect = Math::fresnel_schlick(F0, fabsf(Math::dot(inc_dir, normal)));
        float xi = Math::rnd(seed);

        Vector3float refraction_dir = Math::refract(_eta, inc_dir, normal);
        Vector3float direction;
        if (xi <= p_reflect || Math::safeFloatEqual(Math::norm(refraction_dir), 0.0f))
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
    inline Vector4float
    Material::sample_ggx(uint32_t& seed, const Vector3float& inc_dir, const Vector3float& normal)
    {
        float u = Math::rnd(seed);
        float v = Math::rnd(seed);

        float cosTheta = sqrtf((1.0f - u) / (1.0f + (roughness * roughness - 1.0f) * u));
        float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
        float phi = 2.0f * static_cast<float>(M_PI) * v;

        float x = sinTheta * cosf(phi);
        float y = sinTheta * sinf(phi);
        float z = cosTheta;

        Vector3float H = Math::toLocalFrame(normal, Vector3float(x, y, z));
        Vector3float L = Math::reflect(inc_dir, H);

        float LdotH = Math::dot(inc_dir, H);
        float NdotH = Math::dot(normal, H);

        if (Math::dot(normal, L) <= 0.0f)
        {
            return Vector4float(0, 0, 0, 1);
        }

        float p = detail::D_GGX(NdotH, roughness) * NdotH / fabsf(4.0f * LdotH);

        return Vector4float(L, p);
    }

    __host__ __device__
    inline Vector3float
    Material::brdf_lambert()
    {
        return albedo_d / static_cast<float>(M_PI);
    }

    __host__ __device__
    inline Vector3float
    Material::brdf_phong(const Vector3float& position, const Vector3float& inc_dir, const Vector3float& out_dir, const Vector3float& normal)
    {
        Vector3float halfDir = Math::normalize(inc_dir + out_dir);
        return albedo_s * powf(fmaxf(0.0f, Math::dot(halfDir, normal)), shininess);
    }

    __host__ __device__
    inline Vector3float
    Material::brdf_mirror(const Vector3float& position, const Vector3float& inc_dir, const Vector3float& out_dir, const Vector3float& normal)
    {
        Vector3float reflected = Math::reflect(inc_dir, normal);
        return albedo_s * Math::delta(1.0f - Math::dot(out_dir, reflected)) / Math::dot(out_dir, normal);
    }

    __host__ __device__
    inline Vector3float
    Material::btdf_glass(const Vector3float& position, const Vector3float& inc_dir, const Vector3float& out_dir, const Vector3float& normal)
    {
        if (Math::dot(inc_dir, normal) * Math::dot(out_dir, normal) > 0) //Reflected
        {
            if (Math::dot(inc_dir, normal) > 0)
            {
                return brdf_mirror(position, inc_dir, out_dir, normal);
            }
            else
            {
                return brdf_mirror(position, inc_dir, out_dir, -1.0f * normal);
            }

        }
        else
        {
            Vector3float refracted;
            Vector3float n = normal;
            if (Math::dot(inc_dir, normal) > 0)
            {
                refracted = Math::refract(1.0f / eta, inc_dir, normal);
            }
            else
            {
                refracted = Math::refract(eta, inc_dir, -1.0f * normal);
                n = -1.0f * normal;
            }

            return albedo_s * Math::delta(1.0f - Math::dot(refracted, out_dir)) / Math::dot(inc_dir, n);
        }
    }

    __host__ __device__
    inline Vector3float
    Material::brdf_ggx(const Vector3float& position, const Vector3float& inc_dir, const Vector3float& out_dir, const Vector3float& normal)
    {
        Vector3float H = Math::normalize(inc_dir + out_dir);
        float NdotH = Math::dot(normal, H);
        float LdotH = Math::dot(out_dir, H);
        float NdotL = Math::dot(normal, out_dir);
        float NdotV = Math::dot(normal, inc_dir);

        float ndf = detail::D_GGX(NdotH, roughness);

        float vis = detail::V_SmithJointGGX(NdotL, NdotV, roughness);

        return ndf * vis * Math::fresnel_schlick(albedo_s, LdotH);
    }

    __host__ __device__ 
    inline float 
    Material::henyeyGreensteinPhaseFunction(const float& g, const float& cos_theta)
    {
        if (fabsf(g) < 1e-3f)
            return 0.25 / static_cast<float>(M_PI);

        float g2 = g * g;
        float area = 4.0f * 3.14159f;
        return (1 - g2) / area * powf((1 + g2 - 2.0f * g * cos_theta), -1.5f);
    }

    __host__ __device__ 
    inline Vector4float 
    sampleHenyeyGreensteinPhaseFunction(const float& g_, const Vector3float& forward, uint32_t& seed)
    {
        float g = g_ < 0.0f ? fminf(-1e-3, g_) : fmaxf(1e-3, g_);

        float u1 = Math::rnd(seed);
        float u2 = Math::rnd(seed);

        float g2 = g * g;
        float d = (1.0f - g2) / (1.0f - g + 2.0f * g * u1);
        float cos_theta = Math::clamp(0.5f / g * (1.0f + g2 - d * d), -1.0f, 1.0f);

        float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
        float phi = 2.0f * 3.14159f * u2;

        float x = sin_theta * cosf(phi);
        float y = sin_theta * sinf(phi);
        float z = cos_theta;

        Vector3float result = Math::normalize(Math::toLocalFrame(forward, Vector3float(x, y, z)));

        float pdf = Material::henyeyGreensteinPhaseFunction(g, Math::dot(forward, result));

        return Vector4float(result, pdf);
    }
} //namespace cupbr

#endif