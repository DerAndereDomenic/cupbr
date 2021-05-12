#ifndef __CUPBR_GEOMETRY_MESH_CUH
#define __CUPBR_GEOMETRY_MESH_CUH

#include <Geometry/Geometry.cuh>
#include <Geometry/Triangle.cuh>

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
        */
        __host__ __device__
        Mesh(Triangle* triangle_buffer, const uint32_t& num_triangles);

        //Override
        __host__ __device__
        Vector4float
        computeRayIntersection(const Ray& ray);

        //Override
        __host__ __device__
        Vector3float
        getNormal(const Vector3float& x);
    private:
        Triangle* _triangles;       /**< The plane position */
        Vector3float _normal;       /**< The plane normal */
        uint32_t _num_triangles;    /**< The number of triangles */
};

#include "../../src/Geometry/MeshDetail.cuh"

#endif