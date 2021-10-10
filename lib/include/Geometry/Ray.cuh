#ifndef __CUPBR_GEOMETRY_RAY_CUH
#define __CUPBR_GEOMETRY_RAY_CUH

#include <Math/Vector.h>

namespace cupbr
{
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
        __host__ __device__
            Ray(const Vector3float& origin, const Vector3float direction);

        /**
        *   @brief Get a reference to the ray origin
        *   @return The ray origin
        */
        __host__ __device__
            Vector3float
            origin() const;

        /**
        *   @brief Get a reference to the ray direction
        *   @return The ray direction
        *   @note Normalized
        */
        __host__ __device__
            Vector3float
            direction() const;

        /**
        *   @brief Trace a new ray using the same ray object (This preserves payloads)
        *   @param[in] origin The ray origin
        *   @param[in] direction The ray directions
        *   @note The direction will get normalized
        */
        __host__ __device__
            void
            traceNew(const Vector3float& origin, const Vector3float& direction);

        /**
        *   @brief Set the payload
        *   @tparam PayloadType The payload class stored in this ray
        *   @param[in] payload The new payload
        */
        template<class PayloadType>
        __host__ __device__
            void
            setPayload(PayloadType* payload);

        /**
        *   @brief Get the payload stored in this ray
        *   @tparam PayloadType The payload class stored in this ray
        *   @return Pointer to the payload
        */
        template<class PayloadType>
        __host__ __device__
            PayloadType*
            payload();

    private:
        Vector3float _origin = 0;           /**< The ray origin */
        Vector3float _direction = 0;        /**< The normalized ray direction */
        void* _payload = nullptr;           /**< The ray payload */

    };
} //namespace cupbr

#include "../../src/Geometry/RayDetail.cuh"

#endif