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
        AABB(const Vector3float& minimum, const Vector3float& maximum);

        /**
        *   @brief Check if a ray hits the bounding box
        *   @param[in] ray The ray
        */
        __host__ __device__
        bool hit(const Ray& ray) const;

        /**
        *   @brief Get the minimum point
        *   @return The minimum point
        */
        __host__ __device__
        Vector3float minimum() const;

        /**
        *   @brief Get the maximum point
        *   @return The maximum point
        */
        __host__ __device__
        Vector3float maximum() const;

        private:
        Vector3float _minimum;          /**< The minimum point */
        Vector3float _maximum;          /**< The maximum point */

    };
}

#include "../../src/Geometry/AABBDetail.h"

#endif