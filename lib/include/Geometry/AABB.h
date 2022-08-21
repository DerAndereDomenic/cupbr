#ifndef __CUPBR_GEOMETRY_AABB_H
#define __CUPBR_GEOMETRY_AABB_H

#include <Math/VectorTypes_fwd.h>
#include <Geometry/Ray.h>

namespace cupbr
{
    /**
    *   @brief Class to model an axis aligned bounding box
    */
    class AABB
    {
        public:

        /**
        *   @brief Default constructor
        */
        AABB() = default;

        /**
        *   @brief Constructor
        *   @param[in] minimum The minimum point
        *   @param[in] maximum The maximum point
        */
        CUPBR_HOST_DEVICE
        AABB(const Vector3float& minimum, const Vector3float& maximum);

        /**
        *   @brief Check if a ray hits the bounding box
        *   @param[in] ray The ray
        */
        CUPBR_HOST_DEVICE
        bool hit(const Ray& ray) const;

        /**
        *   @brief Get the minimum point
        *   @return The minimum point
        */
        CUPBR_HOST_DEVICE
        Vector3float minimum() const;

        /**
        *   @brief Get the maximum point
        *   @return The maximum point
        */
        CUPBR_HOST_DEVICE
        Vector3float maximum() const;

        private:
        Vector3float _minimum;          /**< The minimum point */
        Vector3float _maximum;          /**< The maximum point */

    };
}

#include "../../src/Geometry/AABBDetail.h"

#endif