#ifndef __CUPBR_GEOMETRY_PLANEDETAIL_H
#define __CUPBR_GEOMETRY_PLANEDETAIL_H

#include <Core/CUDA.h>
#include <Math/Functions.h>

namespace cupbr
{
    __host__ __device__
    inline
    Plane::Plane(const Vector3float& position, const Vector3float& normal)
        :Geometry(),
        _position(position),
        _normal(Math::normalize(normal))
    {
        type = GeometryType::PLANE;
    }

    __host__ __device__
    inline Vector4float
    Plane::computeRayIntersection(const Ray& ray)
    {
        float PdotN = Math::dot(_position, _normal);
        float OdotN = Math::dot(ray.origin(), _normal);
        float DdotN = Math::dot(ray.direction(), _normal);

        if (Math::safeFloatEqual(DdotN, 0.0f))
        {
            return Vector4float(INFINITY);
        }

        float t = (PdotN - OdotN) / DdotN;

        if (t < 0)return Vector4float(INFINITY);

        return Vector4float(ray.origin() + t * ray.direction(), t);
    }

    __host__ __device__
    inline Vector3float
    Plane::getNormal(const Vector3float& x)
    {
        return _normal;
    }

    __host__ __device__
    inline Vector3float
    Plane::position()
    {
        return _position;
    }
} //namespac cupbr

#endif