#ifndef __CUPBR_GEOMETRY_GEOMETRY_H
#define __CUPBR_GEOMETRY_GEOMETRY_H

#include <Geometry/Geometry.h>
#include <Geometry/Ray.h>
#include <Geometry/Material.h>
#include <Geometry/AABB.h>
#include <cmath>

namespace cupbr
{
    /**
    *   @brief A class to model different geometry types
    */
    enum class GeometryType
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
        Geometry();

        /**
        *   @brief Compute the intersection point of geometry and a ray
        *   @param[in] ray The ray
        *   @return A 4D vector where the first 3 components return the intersection point in world space and the w component encodes the depth to the camera
        *   @note If no intersection was found the vector will be INFINITY
        */
        __host__ __device__
        Vector4float computeRayIntersection(const Ray& ray);

        /**
        *   @brief Get the normal at a specified point
        *   @param[in] x The position in world space
        *   @return The corresponding normal
        *   @note If no normal is found at this point the vector will be INFINITY
        */
        __host__ __device__
        Vector3float getNormal(const Vector3float& x);

        /**
        *   @brief Get the unique geometry id
        *   @return The id 
        */ 
        __host__ __device__
        uint32_t id() const;

        /**
        *   @brief Set the geometry id
        *   @param[in] id The new id 
        */
        inline void setID(const uint32_t& id){ _id = id; }

        Material material = {};                         /**< The material of the object */
        GeometryType type = GeometryType::MESH;         /**< The geometry type */
        protected:
        AABB _aabb;                                     /**< The axis aligned bounding box */
        uint32_t _id;                                   /**< The geometry id */
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

#include "../../src/Geometry/GeometryDetail.h"

#endif