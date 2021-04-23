#ifndef __CUPBR_DATASTRUCTURE_LIGHT_CUH
#define __CUPBR_DATASTRUCTURE_LIGHT_CUH

#include <Math/Vector.h>

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
struct Light
{
    /**
    *   @brief Default constructor 
    */
    Light() = default;

    LightType type = POINT;     /**< The light type */

    Vector3float position;      /**< The light position */
    Vector3float intensity;     /**< The light intensity (point) */
    Vector3float radiance;      /**< The light radiance (area) */
};

#endif