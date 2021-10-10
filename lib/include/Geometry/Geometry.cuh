#ifndef __CUPBR_GEOMETRY_GEOMETRY_CUH
#define __CUPBR_GEOMETRY_GEOMETRY_CUH

#include <Geometry/Geometry.cuh>
#include <Geometry/Ray.cuh>
#include <Geometry/Material.cuh>
#include <cmath>

namespace cupbr
{
    /**
    *   @brief A class to model different geometry types
    */
    enum GeometryType
    {
        SPHERE,
        PLANE,
        QUAD,
        TRIANGLE,
        MESH
    };

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

        Material material;  /**< The material of the object */
        GeometryType type;  /**< The geometry type */
    private:
    };

    /**
    *   @brief A struct to model the local surface geometry
    */
    struct LocalGeometry
    {
        GeometryType type;                  /**< The geometry type */
        float depth = INFINITY;             /**< The depth */
        Vector3float P;                     /**< The intersection point in world space */
        Vector3float N;                     /**< The surface normal */
        Material material;                  /**< The object material */
        int32_t scene_index;                /**< The index in the scene representation */
    };
} //namespace cupbr

#include "../../src/Geometry/GeometryDetail.cuh"

#endif