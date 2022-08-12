#ifndef __CUPBR_GEOMETRY_MATERIALGGXDETAIL_H
#define __CUPBR_GEOMETRY_MATERIALGGXDETAIL_H

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
    MaterialGGX::brdf(const Vector3float& position, const Vector3float& inc_dir, const Vector3float& out_dir, const Vector3float& normal)
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
    inline Vector4float
    MaterialGGX::sampleDirection(uint32_t& seed, const Vector3float& inc_dir, const Vector3float& normal)
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
} //namespace cupbr

#endif