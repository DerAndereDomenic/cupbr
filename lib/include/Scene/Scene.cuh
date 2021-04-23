#ifndef __CUPBR_SCENE_SCENE_CUH
#define __CUPBR_SCENE_SCENE_CUH

#include <Geometry/Geometry.cuh>

/**
*   @brief Struct to model a scene
*/
struct Scene
{
    Geometry** geometry;    /**< The scene geometry */
    uint32_t scene_size;    /**< The number of objects in the scene */

    /**
    *   @brief Get a scene element
    *   @param[in] index The index
    *   @return The geometry 
    */
    __host__ __device__
    Geometry* 
    operator[](const uint32_t index) const;
};

#include "../../src/Scene/SceneDetail.cuh"

#endif