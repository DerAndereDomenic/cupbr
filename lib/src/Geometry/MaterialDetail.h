#ifndef __CUPBR_GEOMETRY_MATERIALDETAIL_H
#define __CUPBR_GEOMETRY_MATERIALDETAIL_H

#include <cmath>
#include <Math/Functions.h>

namespace cupbr
{
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
    Material::sampleHenyeyGreensteinPhaseFunction(const float& g_, const Vector3float& forward, uint32_t& seed)
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