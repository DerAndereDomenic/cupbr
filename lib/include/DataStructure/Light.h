#ifndef __CUPBR_DATASTRUCTURE_LIGHT_H
#define __CUPBR_DATASTRUCTURE_LIGHT_H

#include <Math/Vector.h>

namespace cupbr
{
    /**
    *   @brief Model the light type
    */
    enum LightType
    {
        POINT,      /**< A point light source */
        AREA        /**< A area light source */
    };

    /**
    *   @brief A struct to model a light source
    */
    class Light
    {
        public:
        /**
        *   @brief Default constructor
        */
        Light() = default;

        LightType type = POINT;     /**< The light type */

        Vector3float position;      /**< The light position */
        Vector3float intensity;     /**< The light intensity (point) */
        Vector3float radiance;      /**< The light radiance (area) */
        Vector3float halfExtend1;   /**< The first half extend in world space (area) */
        Vector3float halfExtend2;   /**< The second half extend in world space (area) */

        /**
        *   @brief Sample the light source
        *   @param[in] seed The seed for random number generation
        *   @param[in] position The sample position
        *   @param[out] lightDir The light direction
        *   @param[out] distance The distance to the sample point
        *   @return The radiance received by the given position
        */
        __host__ __device__
        Vector3float sample(uint32_t& seed, const Vector3float& position, Vector3float& lightDir, float& distance);

        private:

        /**
        *   @brief Sample the point light source
        *   @param[in] seed The seed for random number generation
        *   @param[in] position The sample position
        *   @param[out] lightDir The light direction
        *   @param[out] distance The distance to the sample point
        *   @return The radiance received by the given position
        */
        __host__ __device__
        Vector3float sample_point(uint32_t& seed, const Vector3float& position, Vector3float& lightDir, float& distance);

        /**
        *   @brief Sample the area light source
        *   @param[in] seed The seed for random number generation
        *   @param[in] position The sample position
        *   @param[out] lightDir The light direction
        *   @param[out] distance The distance to the sample point
        *   @return The radiance received by the given position
        */
        __host__ __device__
        Vector3float sample_area(uint32_t& seed, const Vector3float& position, Vector3float& lightDir, float& distance);

    };
} //namespace cupbr

#include "../../src/DataStructure/LightDetail.h"

#endif