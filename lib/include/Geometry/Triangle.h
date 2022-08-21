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
        CUPBR_HOST_DEVICE
        Triangle(const Vector3float& vertex1,
                 const Vector3float& vertex2,
                 const Vector3float& vertex3);

        /**
        *   @brief Create a Triangle
        *   @param[in] vertex1 The first vertex
        *   @param[in] vertex2 The second vertex
        *   @param[in] vertex3 The third vertex
        *   @param[in] normal1 The first normal
        *   @param[in] normal2 The second normal
        *   @param[in] normal3 The third normal
        *   @note The normal will get normalized
        */
        CUPBR_HOST_DEVICE
        Triangle(const Vector3float& vertex1,
                 const Vector3float& vertex2,
                 const Vector3float& vertex3,
                 const Vector3float& normal1,
                 const Vector3float& normal2,
                 const Vector3float& normal3);
        
        //Override
        CUPBR_HOST_DEVICE
        LocalGeometry computeRayIntersection(const Ray& ray);

        private:
        Vector3float _vertex1;     /**< The first vertex */
        Vector3float _vertex2;     /**< The second vertex */
        Vector3float _vertex3;     /**< The third vertex */
        Vector3float _normal1;     /**< The first normal */
        Vector3float _normal2;     /**< The second normal */
        Vector3float _normal3;     /**< The third normal */
    };

} //namespace cupbr

#include "../../src/Geometry/TriangleDetail.h"

#endif