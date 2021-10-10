#ifndef __CUPBR_SCENE_SCENE_CUH
#define __CUPBR_SCENE_SCENE_CUH

#include <Geometry/Geometry.cuh>
#include <DataStructure/Light.cuh>
#include <DataStructure/Image.cuh>

namespace cupbr
{
    /**
    *   @brief A struct to model a volume
    */
    struct Volume
    {
        float sigma_s = 1.0f;      /**< The scattering coefficient */
        float sigma_a = 1.0f;      /**< The absorbtion coefficient */
        float g = 0.0f;            /**< The phase function parameter */
    };

    /**
    *   @brief Struct to model a scene
    */
    struct Scene
    {
        Geometry** geometry;                /**< The scene geometry */
        uint32_t scene_size;                /**< The number of objects in the scene */
        Light** lights;                     /**< The light sources in the scene */
        uint32_t light_count;               /**< The light source count */
        bool useEnvironmentMap = false;     /**< Wether an environment map is loaded */
        Image<Vector3float> environment;    /**< The Environment map */
        Volume volume;                      /**< The volume inside the scene */

        /**
        *   @brief Get a scene element
        *   @param[in] index The index
        *   @return The geometry
        */
        __host__ __device__
            Geometry*
            operator[](const uint32_t index) const;
    };
} //namespace cupbr

#include "../../src/Scene/SceneDetail.cuh"

#endif