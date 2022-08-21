#ifndef __CUPBR_GEOMETRY_QUADDETAIL_H
#define __CUPBR_GEOMETRY_QUADDETAIL_H

#include <Core/CUDA.h>
#include <Math/Functions.h>

namespace cupbr
{
    CUPBR_HOST_DEVICE
    inline
    Quad::Quad(const Vector3float& position, const Vector3float& normal, const Vector3float& extend1, const Vector3float& extend2)
        :Geometry(),
        _position(position),
        _normal(Math::normalize(normal)),
        _extend1(extend1),
        _extend2(extend2)
    {
        type = GeometryType::QUAD;
    }

    CUPBR_HOST_DEVICE
    inline LocalGeometry
    Quad::computeRayIntersection(const Ray& ray)
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

        Vector3float intersection = ray.origin() + t * ray.direction();

        float x = Math::dot(intersection - _position, _extend1);
        float y = Math::dot(intersection - _position, _extend2);

        float side1 = Math::dot(_extend1, _extend1);
        float side2 = Math::dot(_extend2, _extend2);

        if (x < -side1 || x > side1 || y < -side2 || y > side2)return geom;

        geom.type = GeometryType::QUAD;
        geom.P = intersection;
        geom.depth = t;
        geom.N = _normal;
        geom.material = material;
        geom.scene_index = _id;

        return geom;
    }

    CUPBR_HOST_DEVICE
    inline Vector3float
    Quad::position()
    {
        return _position;
    }
} //namespace cupbr

#endif