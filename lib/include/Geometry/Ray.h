#ifndef __CUPBR_GEOMETRY_RAY_H
#define __CUPBR_GEOMETRY_RAY_H

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
        CUPBR_HOST_DEVICE
        Ray(const Vector3float& origin, const Vector3float direction);

        /**
        *   @brief Get a reference to the ray origin
        *   @return The ray origin
        */
        CUPBR_HOST_DEVICE
        Vector3float origin() const;

        /**
        *   @brief Get a reference to the ray direction
        *   @return The ray direction
        *   @note Normalized
        */
        CUPBR_HOST_DEVICE
        Vector3float direction() const;

        /**
        *   @brief Trace a new ray using the same ray object (This preserves payloads)
        *   @param[in] origin The ray origin
        *   @param[in] direction The ray directions
        *   @note The direction will get normalized
        */
        CUPBR_HOST_DEVICE
        void traceNew(const Vector3float& origin, const Vector3float& direction);

        /**
        *   @brief Set the payload
        *   @tparam PayloadType The payload class stored in this ray
        *   @param[in] payload The new payload
        */
        template<class PayloadType>
        CUPBR_HOST_DEVICE
        void setPayload(PayloadType* payload);

        /**
        *   @brief Get the payload stored in this ray
        *   @tparam PayloadType The payload class stored in this ray
        *   @return Pointer to the payload
        */
        template<class PayloadType>
        CUPBR_HOST_DEVICE
        PayloadType* payload();

        private:
        Vector3float _origin = 0;           /**< The ray origin */
        Vector3float _direction = 0;        /**< The normalized ray direction */
        void* _payload = nullptr;           /**< The ray payload */

    };
} //namespace cupbr

#include "../../src/Geometry/RayDetail.h"

#endif