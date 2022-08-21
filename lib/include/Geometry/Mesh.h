#ifndef __CUPBR_GEOMETRY_MESH_H
#define __CUPBR_GEOMETRY_MESH_H

#include <Geometry/Geometry.h>
#include <Geometry/Triangle.h>

namespace cupbr
{
    /**
    *   @brief A class to model a Mesh
    */
    class Mesh : public Geometry
    {
        public:
        /**
        *   @brief Default constructor
        */
        Mesh();

        /**
        *   @brief Create a mesh
        *   @param[in] triangle_buffer The triangle buffer
        *   @param[in] num_triangles The number of triangles
        *   @param[in] minimum The minimum point of the aabb
        *   @�aram[in] maximum The maximum point of the aabb
        */
        CUPBR_HOST_DEVICE
        Mesh(Triangle* triangle_buffer, const uint32_t& num_triangles, const Vector3float& minimum, const Vector3float& maximum);

        //Override
        CUPBR_HOST_DEVICE
        LocalGeometry computeRayIntersection(const Ray& ray);

        /**
        *   @brief Get the number of triangles
        *   @return The number of triangles
        */
        CUPBR_HOST_DEVICE
        uint32_t num_triangles();

        CUPBR_HOST_DEVICE
        Triangle* triangles();

        private:
        Triangle* _triangles;       /**< The plane position */
        uint32_t _num_triangles;    /**< The number of triangles */
    };

} //namespace cupbr

#include "../../src/Geometry/MeshDetail.h"

#endif