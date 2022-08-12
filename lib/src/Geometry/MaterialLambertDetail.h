#ifndef __CUPBR_GEOMETRY_MATERIALLAMBERTDETAIL_H
#define __CUPBR_GEOMETRY_MATERIALLAMBERTDETAIL_H

#include <cmath>
#include <Math/Functions.h>

namespace cupbr
{
    __host__ __device__
    inline Vector3float
    MaterialLambert::brdf(const Vector3float& position, const Vector3float& inc_dir, const Vector3float& out_dir, const Vector3float& normal)
    {
        return albedo_d / static_cast<float>(M_PI);
    }

    __host__ __device__
    inline Vector4float
    MaterialLambert::sampleDirection(uint32_t& seed, const Vector3float& inc_dir, const Vector3float& normal)
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
} //namespace cupbr

#endif