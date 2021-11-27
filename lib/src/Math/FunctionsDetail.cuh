#ifndef __CUPBR_MATH_FUNCTIONSDETAIL_CUH
#define __CUPBR_MATH_FUNCTIONSDETAIL_CUH

#include <Core/CUDA.cuh>
#include <cmath>
#include <algorithm>

namespace cupbr
{
    __host__ __device__
    inline bool
    Math::safeFloatEqual(const float& lhs, const float& rhs, const float& eps)
    {
        return std::abs(lhs - rhs) < eps;
    }

    template<typename T>
    T
    Math::clamp(const T& x, const T& mini, const T& maxi)
    {
        return fminf(fmaxf(x, mini), maxi);
    }

    __host__ __device__
    inline float
    Math::delta(const float& inp)
    {
        return safeFloatEqual(inp, 0.0f) ? 1.0f : 0.0f;
    }

    __host__ __device__
    inline Vector3float
    Math::reflect(const Vector3float& inc_dir, const Vector3float& normal)
    {
        return normalize(2.0f * dot(inc_dir, normal) * normal - inc_dir);
    }

    __host__ __device__
    inline Vector3float
    Math::refract(const float& eta, const Vector3float& inc_dir, const Vector3float& normal)
    {
        float NdotI = dot(inc_dir, normal);
        float k = 1.0f - eta * eta * (1.0f - NdotI * NdotI);
        if (k < 0)
        {
            return 0;
        }
        else
        {
            return -eta * inc_dir - (-eta * NdotI + sqrtf(k)) * normal;
        }
    }

    __host__ __device__
    inline float
    Math::fresnel_schlick(const float& F0, const float& VdotH)
    {
        float p = fmaxf(0.0f, (1.0f - VdotH));
        return F0 + (1.0f - F0) * p * p * p * p * p;
    }

    __host__ __device__
    inline Vector3float
    Math::fresnel_schlick(const Vector3float& F0, const float& VdotH)
    {
        float p = fmaxf(0.0f, (1.0f - VdotH));
        return F0 + (Vector3float(1.0f) - F0) * p * p * p * p * p; //powf seems to be instable in fast math mode
    }

    template<uint32_t N>
    __host__ __device__
    uint32_t
    Math::tea(uint32_t val0, uint32_t val1)
    {
        uint32_t v0 = val0;
        uint32_t v1 = val1;
        uint32_t s0 = 0;

        for (uint32_t n = 0; n < N; ++n)
        {
            s0 += 0x9e3779b9;
            v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
            v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
        }

        return v0;
    }

    __host__ __device__
    inline float
    Math::rnd(uint32_t& prev)
    {
        const uint32_t LCG_A = 1664525u;
        const uint32_t LCG_C = 1013904223u;
        prev = (LCG_A * prev + LCG_C);

        return ((float)(prev & 0x00FFFFFF) / (float)0x01000000);
    }

    __host__ __device__
    inline Vector3float
    Math::toLocalFrame(const Vector3float& N, const Vector3float& direction)
    {
        const float x = N.x;
        const float y = N.y;
        const float z = N.z;
        const float sz = (z >= 0.0f) ? 1.0f : -1.0f;
        const float a = 1.0f / (sz + z);
        const float ya = y * a;
        const float b = x * ya;
        const float c = x * sz;

        Vector3float localX(c * x * a - 1.0f, sz * b, c);
        Vector3float localY(b, y * ya - sz, y);

        return direction.x * localX + direction.y * localY + direction.z * N;
    }

} //namespace cupbr

#endif
