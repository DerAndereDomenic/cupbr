#ifndef __CUPPBR_DATASTRUCTURE_LIGHTDETAIL_CUH
#define __CUPPBR_DATASTRUCTURE_LIGHTDETAIL_CUH

#include <Math/Vector.h>

namespace cupbr
{
    __host__ __device__
    inline Vector3float
    Light::sample(uint32_t& seed, const Vector3float& position, Vector3float& lightDir, float& distance)
    {
        switch (type)
        {
            case POINT:
                return sample_point(seed, position, lightDir, distance);
            case AREA:
                return sample_area(seed, position, lightDir, distance);
        }

        return Vector3float(0);
    }

    __host__ __device__
    inline Vector3float
    Light::sample_point(uint32_t& seed, const Vector3float& position, Vector3float& lightDir, float& distance)
    {
        lightDir = Math::normalize(this->position - position);
        distance = Math::norm(this->position - position);
        return intensity / (distance * distance);
    }

    __host__ __device__
    inline Vector3float
    Light::sample_area(uint32_t& seed, const Vector3float& position, Vector3float& lightDir, float& distance)
    {
        float xi1 = Math::rnd(seed) * 2.0f - 1.0f;
        float xi2 = Math::rnd(seed) * 2.0f - 1.0f;

        Vector3float sample = this->position + xi1 * halfExtend1 + xi2 * halfExtend2;
        Vector3float n = Math::normalize(Math::cross(halfExtend1, halfExtend2));
        float area = 4.0f * Math::norm(halfExtend1) * Math::norm(halfExtend2);

        lightDir = Math::normalize(sample - position);
        distance = Math::norm(sample - position);

        float NdotL = Math::dot(lightDir, n);
        if (NdotL < 0) NdotL *= -1.0f;

        float solidAngle = area * NdotL / (distance * distance);

        return radiance * solidAngle;
    }

} //namespace cupbr

#endif