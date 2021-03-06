#ifndef __CUPBR_GEOMETRY_TRIANGLE_H
#define __CUPBR_GEOMETRY_TRIANGLE_H

#include <Geometry/Geometry.h>

namespace cupbr
{
    /**
    *   @brief A class to model a Triangle
    */
    class Triangle : public Geometry
    {
        public:
        /**
        *   @brief Default constructor
        */
        Triangle();

        /**
        *   @brief Create a Triangle
        *   @param[in] vertex1 The first vertex
        *   @param[in] vertex2 The second vertex
        *   @param[in] vertex3 The third vertex
        *   @note The normal will get normalized
        */
        __host__ __device__
        Triangle(const Vector3float& vertex1, const Vector3float& vertex2, const Vector3float& vertex3);

        //Override
        __host__ __device__
        LocalGeometry computeRayIntersection(const Ray& ray);

        private:
        Vector3float _vertex1;     /**< The first vertex */
        Vector3float _vertex2;     /**< The second vertex */
        Vector3float _vertex3;     /**< The third vertex */
        Vector3float _normal;      /**< The normal */
    };

} //namespace cupbr

#include "../../src/Geometry/TriangleDetail.h"

#endif