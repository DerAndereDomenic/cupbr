#ifndef __CUPBR_GEOMETRY_TRIANGLEDETAIL_CUH
#define __CUPBR_GEOMETRY_TRIANGLEDETAIL_CUH

#include <Core/CUDA.cuh>
#include <Math/Functions.cuh>

namespace cupbr
{
    __host__ __device__
    inline
    Triangle::Triangle(const Vector3float& vertex1, const Vector3float& vertex2, const Vector3float& vertex3)
        :_vertex1(vertex1),
        _vertex2(vertex2),
        _vertex3(vertex3),
        _normal(Math::normalize(Math::cross(vertex2 - vertex1, vertex3 - vertex1)))
    {
        type = GeometryType::TRIANGLE;
    }

    __host__ __device__
    inline Vector4float
    Triangle::computeRayIntersection(const Ray& ray)
    {
        Vector3float v0v1 = _vertex2 - _vertex1;
        Vector3float v0v2 = _vertex3 - _vertex1;
        Vector3float pvec = Math::cross(ray.direction(), v0v2);
        float det = Math::dot(v0v1, pvec);

        if (Math::safeFloatEqual(det, 0.0f))return Vector4float(INFINITY);

        float invDet = 1.0f / det;

        Vector3float tvec = ray.origin() - _vertex1;

        float u = Math::dot(tvec, pvec) * invDet;
        if (u < 0.0f || u > 1.0f) return Vector4float(INFINITY);

        Vector3float qvec = Math::cross(tvec, v0v1);
        float v = Math::dot(ray.direction(), qvec) * invDet;
        if (v < 0.0f || u + v > 1.0f) return Vector4float(INFINITY);

        float t = Math::dot(v0v2, qvec) * invDet;

        if (t < 0.0f) return Vector4float(INFINITY);

        return Vector4float(ray.origin() + t * ray.direction(), t);
    }

    __host__ __device__
    inline Vector3float
    Triangle::getNormal(const Vector3float& x)
    {
        return _normal;
    }
} //namespace cupbr

#endif