#ifndef __CUPBR_GEOMETRY_QUAD_H
#define __CUPBR_GEOMETRY_QUAD_H

#include <Geometry/Geometry.h>

namespace cupbr
{
    /**
    *   @brief A class to model a Quad
    */
    class Quad : public Geometry
    {
        public:
        /**
        *   @brief Default constructor
        */
        Quad();

        /**
        *   @brief Create a Quad
        *   @param[in] position The center position on the quad
        *   @param[in] normal The quad normal
        *   @param[in] extend1 The first extend
        *   @param[in] extend2 The second extend
        *   @note The normal will get normalized
        */
        CUPBR_HOST_DEVICE
        Quad(const Vector3float& position, const Vector3float& normal, const Vector3float& extend1, const Vector3float& extend2);

        //Override
        CUPBR_HOST_DEVICE
        LocalGeometry computeRayIntersection(const Ray& ray);

        /**
        *   @brief Get the position of the Quad
        *   @return The position to define the quad
        */
        CUPBR_HOST_DEVICE
        Vector3float position();

        private:
        Vector3float _position;     /**< The plane position */
        Vector3float _normal;       /**< The plane normal */
        Vector3float _extend1;      /**< The extend in the first direction */
        Vector3float _extend2;      /**< The extend in the second direction */
    };
} //namespace cupbr

#include "../../src/Geometry/QuadDetail.h"

#endif