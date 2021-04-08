#ifndef __CUPBR_GEOMETRY_RAY_CUH
#define __CUPBR_GEOMETRY_RAY_CUH

#include <Math/Vector.h>

/**
*   @brief Class to model a ray
*/
class Ray
{
    public:
        /**
        *   @brief Default constructor
        */
        Ray() = default;

        /**
        *   @brief Create a ray
        *   @param[in] origin The ray origin
        *   @param[in] direction The ray direction
        *   @note The direction will get normalized by the constructor
        */
        Ray(const Vector3float& origin, const Vector3float direction);

        /**
        *   @brief Get a reference to the ray origin
        *   @return The ray origin
        */
        __host__ __device__
        Vector3float&
        origin();

        /**
        *   @brief Get a reference to the ray direction
        *   @return The ray direction
        *   @note Normalized
        */
        __host__ __device__
        Vector3float&
        direction();

    private:
        Vector3float _origin;       /**< The ray origin */
        Vector3float _direction;    /**< The normalized ray direction */
};

#include "../../src/Geometry/RayDetail.cuh"

#endif