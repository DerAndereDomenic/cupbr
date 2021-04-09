#ifndef __CUPBR_GEOMETRY_GEOMETRY_CUH
#define __CUPBR_GEOMETRY_GEOMETRY_CUH

#include <Geometry/Geometry.cuh>
#include <Geometry/Ray.cuh>
#include <cmath>

/**
*   @brief Model scene geometry
*/
class Geometry
{
    public:
        /**
        *   @brief Default constructor
        */
        Geometry() = default;

        /**
        *   @brief Compute the intersection point of geometry and a ray
        *   @param[in] ray The ray
        *   @return A 4D vector where the first 3 components return the intersection point in world space and the w component encodes the depth to the camera
        *   @note If no intersection was found the vector will be INFINITY
        */
        __host__ __device__
        Vector4float
        computeRayIntersection(const Ray& ray);

        /**
        *   @brief Get the normal at a specified point
        *   @param[in] x The position in world space
        *   @return The corresponding normal
        *   @note If no normal is found at this point the vector will be INFINITY
        */
        __host__ __device__
        Vector3float
        getNormal(const Vector3float& x);
    private:
};

#include "../../src/Geometry/GeometryDetail.cuh"

#endif