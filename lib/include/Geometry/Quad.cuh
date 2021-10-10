#ifndef __CUPBR_GEOMETRY_QUAD_CUH
#define __CUPBR_GEOMETRY_QUAD_CUH

#include <Geometry/Geometry.cuh>

namespace cupbr
{
    /**
    *   @brief A class to model a Quad
    */
    class Quad : public Geometry
    {
    public:
        /**
        *   @brief Default constructor
        */
        Quad();

        /**
        *   @brief Create a Quad
        *   @param[in] position The center position on the quad
        *   @param[in] normal The quad normal
        *   @param[in] extend1 The first extend
        *   @param[in] extend2 The second extend
        *   @note The normal will get normalized
        */
        __host__ __device__
            Quad(const Vector3float& position, const Vector3float& normal, const Vector3float& extend1, const Vector3float& extend2);

        //Override
        __host__ __device__
            Vector4float
            computeRayIntersection(const Ray& ray);

        //Override
        __host__ __device__
            Vector3float
            getNormal(const Vector3float& x);

        /**
        *   @brief Get the position of the Quad
        *   @return The position to define the quad
        */
        __host__ __device__
            Vector3float
            position();
    private:
        Vector3float _position;     /**< The plane position */
        Vector3float _normal;       /**< The plane normal */
        Vector3float _extend1;      /**< The extend in the first direction */
        Vector3float _extend2;      /**< The extend in the second direction */
    };
} //namespace cupbr

#include "../../src/Geometry/QuadDetail.cuh"

#endif