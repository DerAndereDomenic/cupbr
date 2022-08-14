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
        MESH,
        BVH
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
        Material* material;                 /**< The object material */
        int32_t scene_index;                /**< The index in the scene representation */
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
        __host__ __device__
        Geometry::Geometry()
        :_aabb(AABB(-INFINITY, INFINITY)){}

        /**
        *   @brief Compute the intersection point of geometry and a ray
        *   @param[in] ray The ray
        *   @return The local geometry information
        *   @note If no intersection was found the depth will be INFINITY
        */
        __host__ __device__
        LocalGeometry computeRayIntersection(const Ray& ray);

        /**
        *   @brief Get the unique geometry id
        *   @return The id 
        */ 
        __host__ __device__
        uint32_t id() const;

        /**
        *   @brief Get the aabb
        *   @return The aabb
        */
        __host__ __device__
        AABB aabb() const;

        /**
        *   @brief Set the geometry id
        *   @param[in] id The new id 
        */
        inline void setID(const uint32_t& id){ _id = id; }

        Material* material = nullptr;                   /**< The material of the object */
        GeometryType type = GeometryType::MESH;         /**< The geometry type */
        protected:
        AABB _aabb;         /**< The axis aligned bounding box */
        uint32_t _id;                                   /**< The geometry id */
    };

} //namespace cupbr

#include "../../src/Geometry/GeometryDetail.h"

#endif