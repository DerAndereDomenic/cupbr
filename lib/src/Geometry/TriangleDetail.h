#ifndef __CUPBR_GEOMETRY_TRIANGLEDETAIL_H
#define __CUPBR_GEOMETRY_TRIANGLEDETAIL_H

#include <Core/CUDA.h>
#include <Math/Functions.h>

namespace cupbr
{
    __host__ __device__
    inline
    Triangle::Triangle(const Vector3float& vertex1,
                       const Vector3float& vertex2,
                       const Vector3float& vertex3)
        :Geometry(),
        _vertex1(vertex1),
        _vertex2(vertex2),
        _vertex3(vertex3),
        _normal1(Math::normalize(Math::cross(vertex2 - vertex1, vertex3 - vertex1))),
        _normal2(Math::normalize(Math::cross(vertex2 - vertex1, vertex3 - vertex1))),
        _normal3(Math::normalize(Math::cross(vertex2 - vertex1, vertex3 - vertex1)))
    {
        type = GeometryType::TRIANGLE;
    }

    __host__ __device__
    inline
    Triangle::Triangle(const Vector3float& vertex1,
                       const Vector3float& vertex2,
                       const Vector3float& vertex3,
                       const Vector3float& normal1,
                       const Vector3float& normal2,
                       const Vector3float& normal3)
        :Geometry(),
        _vertex1(vertex1),
        _vertex2(vertex2),
        _vertex3(vertex3),
        _normal1(normal1),
        _normal2(normal2),
        _normal3(normal3)
    {
        type = GeometryType::TRIANGLE;
    }

    __host__ __device__
    inline LocalGeometry
    Triangle::computeRayIntersection(const Ray& ray)
    {
        LocalGeometry geom;

        Vector3float v0v1 = _vertex2 - _vertex1;
        Vector3float v0v2 = _vertex3 - _vertex1;
        Vector3float pvec = Math::cross(ray.direction(), v0v2);
        float det = Math::dot(v0v1, pvec);

        if (Math::safeFloatEqual(det, 0.0f))return geom;

        float invDet = 1.0f / det;

        Vector3float tvec = ray.origin() - _vertex1;

        float u = Math::dot(tvec, pvec) * invDet;
        if (u < 0.0f || u > 1.0f) return geom;

        Vector3float qvec = Math::cross(tvec, v0v1);
        float v = Math::dot(ray.direction(), qvec) * invDet;
        if (v < 0.0f || u + v > 1.0f) return geom;

        float t = Math::dot(v0v2, qvec) * invDet;

        if (t < 0.0f) return geom;

        float w = 1.0f - u - v;

        geom.type = GeometryType::TRIANGLE;
        geom.P = ray.origin() + t * ray.direction();
        geom.depth = t;
        geom.N = Math::normalize(w * _normal1 + u * _normal2 + v * _normal3);
        geom.material = material;
        geom.scene_index = _id;

        return geom;
    }
} //namespace cupbr

#endif