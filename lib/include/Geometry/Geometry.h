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
        CUPBR_HOST_DEVICE
        Geometry()
        :_aabb(AABB(-INFINITY, INFINITY)){}

        /**
        *   @brief Compute the intersection point of geometry and a ray
        *   @param[in] ray The ray
        *   @return The local geometry information
        *   @note If no intersection was found the depth will be INFINITY
        */
        CUPBR_HOST_DEVICE
        LocalGeometry computeRayIntersection(const Ray& ray);

        /**
        *   @brief Get the unique geometry id
        *   @return The id 
        */ 
        CUPBR_HOST_DEVICE
        uint32_t id() const;

        /**
        *   @brief Get the aabb
        *   @return The aabb
        */
        CUPBR_HOST_DEVICE
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