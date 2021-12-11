#ifndef __CUPBR_GEOMETRY_SPHERE_H
#define __CUPBR_GEOMETRY_SPHERE_H

#include <Geometry/Geometry.h>

namespace cupbr
{
    /**
    *   @brief A class to model a sphere
    */
    class Sphere : public Geometry
    {
        public:
        /**
        *   @brief Default constructor
        */
        Sphere();

        /**
        *   @brief Creates a sphere
        *   @param[in] position The center position in world space
        *   @param[in] radius The radius
        */
        __host__ __device__
        Sphere(const Vector3float& position, const float& radius);

        /**
        *   @brief Get the sphere position in world space
        *   @return The 3D world position
        */
        __host__ __device__
        Vector3float position() const;

        /**
        *   @brief Get the sphere radius
        *   @return The radius
        */
        __host__ __device__
        float radius() const;

        //Override
        __host__ __device__
        Vector4float computeRayIntersection(const Ray& ray);

        //Override
        __host__ __device__
        Vector3float getNormal(const Vector3float& x);

        private:
        Vector3float _position;     /**< 3D world position of the sphere */
        float _radius;              /**< The radius */
    };
} //namespace cupbr

#include "../../src/Geometry/SphereDetail.h"

#endif