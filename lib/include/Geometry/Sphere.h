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
        CUPBR_HOST_DEVICE
        Sphere(const Vector3float& position, const float& radius);

        /**
        *   @brief Get the sphere position in world space
        *   @return The 3D world position
        */
        CUPBR_HOST_DEVICE
        Vector3float position() const;

        /**
        *   @brief Get the sphere radius
        *   @return The radius
        */
        CUPBR_HOST_DEVICE
        float radius() const;

        //Override
        CUPBR_HOST_DEVICE
        LocalGeometry computeRayIntersection(const Ray& ray);

        private:
        Vector3float _position;     /**< 3D world position of the sphere */
        float _radius;              /**< The radius */
    };
} //namespace cupbr

#include "../../src/Geometry/SphereDetail.h"

#endif