#ifndef __CUPBR_GEOMETRY_MATERIALMIRRORDETAIL_H
#define __CUPBR_GEOMETRY_MATERIALMIRRORDETAIL_H

#include <cmath>
#include <Math/Functions.h>

namespace cupbr
{
    __host__ __device__
    inline Vector3float
    MaterialMirror::brdf(const Vector3float& position, const Vector3float& inc_dir, const Vector3float& out_dir, const Vector3float& normal)
    {
        Vector3float reflected = Math::reflect(inc_dir, normal);
        return albedo_s * Math::delta(1.0f - Math::dot(out_dir, reflected)) / Math::dot(out_dir, normal);
    }

    __host__ __device__
    inline Vector4float
    MaterialMirror::sampleDirection(uint32_t& seed, const Vector3float& inc_dir, const Vector3float& normal)
    {
        Vector3float direction = Math::reflect(inc_dir, normal);

        return Vector4float(direction, 1.0f);
    }
} //namespace cupbr

#endif