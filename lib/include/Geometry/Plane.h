#ifndef __CUPBR_GEOMETRY_PLANE_H
#define __CUPBR_GEOMETRY_PLANE_H

#include <Geometry/Geometry.h>

namespace cupbr
{
    /**
    *   @brief A class to model a Plane
    */
    class Plane : public Geometry
    {
        public:
        /**
        *   @brief Default constructor
        */
        Plane();

        /**
        *   @brief Create a plane
        *   @param[in] position A position on the plane
        *   @param[in] normal The plane normal
        *   @note The normal will get normalized
        */
        __host__ __device__
        Plane(const Vector3float& position, const Vector3float& normal);

        //Override
        __host__ __device__
        LocalGeometry computeRayIntersection(const Ray& ray);

        /**
        *   @brief Get the position of the plane
        *   @return The position to define the plane
        */
        __host__ __device__
        Vector3float position();

        private:
        Vector3float _position;     /**< The plane position */
        Vector3float _normal;       /**< The plane normal */
    };
} //namespace cupbr

#include "../../src/Geometry/PlaneDetail.h"

#endif