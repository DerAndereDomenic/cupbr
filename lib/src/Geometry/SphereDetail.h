#ifndef __CUPBR_GEOMETRY_SPHEREDETAIL_H
#define __CUPBR_GEOMETRY_SPHEREDETAIL_H

#include <Core/CUDA.h>

namespace cupbr
{
    CUPBR_HOST_DEVICE
    inline
    Sphere::Sphere(const Vector3float& position, const float& radius)
        :Geometry(),
        _position(position),
        _radius(radius)
    {
        type = GeometryType::SPHERE;
        _aabb = AABB(position - Vector3float(radius), position + Vector3float(radius));
    }

    CUPBR_HOST_DEVICE
    inline Vector3float
    Sphere::position() const
    {
        return _position;
    }

    CUPBR_HOST_DEVICE
    inline float
    Sphere::radius() const
    {
        return _radius;
    }

    CUPBR_HOST_DEVICE
    inline LocalGeometry
    Sphere::computeRayIntersection(const Ray& ray)
    {
        LocalGeometry geom;

        const Vector3float origin = ray.origin();
        const Vector3float direction = ray.direction();

        Vector3float OS = origin - _position;
        float b = 2.0f * Math::dot(direction, OS);
        float c = Math::dot(OS, OS) - _radius * _radius;
        float disc = b * b - 4 * c;

        if (disc > 0)
        {
            float distSqrt = sqrtf(disc);
            float q = b < 0 ? (-b - distSqrt) / 2.0f : (-b + distSqrt) / 2.0f;
            float t0 = q;
            float t1 = c / q;

            float min = fminf(t0, t1);
            float max = fmaxf(t0, t1);
            t0 = min;
            t1 = max;

            float t = t0 < 0 ? t1 : t0;

            if (t >= 0)
            {
                geom.type = GeometryType::SPHERE;
                geom.P = origin + t * direction;
                geom.depth = t;
                geom.N = Math::normalize(geom.P - _position);
                geom.material = material;
                geom.scene_index = _id;

                return geom;
            }
        }
        return geom;
    }
} //namespace cupbr

#endif