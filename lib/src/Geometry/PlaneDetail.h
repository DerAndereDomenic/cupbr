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
    inline LocalGeometry
    Plane::computeRayIntersection(const Ray& ray)
    {
        LocalGeometry geom;

        float PdotN = Math::dot(_position, _normal);
        float OdotN = Math::dot(ray.origin(), _normal);
        float DdotN = Math::dot(ray.direction(), _normal);

        if (Math::safeFloatEqual(DdotN, 0.0f))
        {
            return geom;
        }

        float t = (PdotN - OdotN) / DdotN;

        if (t < 0)return geom;

        geom.type = GeometryType::PLANE;
        geom.P = ray.origin() + t * ray.direction();
        geom.depth = t;
        geom.material = material;
        geom.N = _normal;
        geom.scene_index = _id;

        return geom;
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