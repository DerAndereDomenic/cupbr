#ifndef __CUPBR_GEOMETRY_PLANE_CUH
#define __CUPBR_GEOMETRY_PLANE_CUH

#include <Geometry/Geometry.cuh>

/**
*   @brief A class to model a Plane 
*/
class Plane : public Geometry
{
    public:
        /**
        *   @brief Default constructor 
        */ 
        Plane() = default;

        /**
        *   @brief Create a plane
        *   @param[in] position A position on the plane
        *   @param[in] normal The plane normal
        */
        __host__ __device__
        Plane(const Vector3float& position, const Vector3float& normal);

        //Override
        __host__ __device__
        Vector4float
        computeRayIntersection(const Ray& ray);

        //Override
        __host__ __device__
        Vector3float
        getNormal(const Vector3float& x);

        /**
        *   @brief Get the position of the plane
        *   @return The position to define the plane 
        */
        __host__ __device__
        Vector3float
        position();
    private:
        Vector3float _position;     /**< The plane position */
        Vector3float _normal;       /**< The plane normal */
};

#endif