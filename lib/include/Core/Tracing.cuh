#ifndef __CUPBR_CORE_TRACING_CUH
#define __CUPBR_CORE_TRACING_CUH

#include <Core/CUDA.cuh>
#include <Geometry/Ray.cuh>
#include <Geometry/Geometry.cuh>
#include <DataStructure/Camera.cuh>
#include <Scene/Scene.cuh>

/**
*   @brief A namespace that capsulate important ray tracing methods
*/
namespace Tracing
{
    /**
    *   @brief Maps a kernel thread index to an output ray
    *   @param[in] tid The kernel thread index
    *   @param[in] width The width of the output image
    *   @param[in] height The height of the output image
    *   @param[in] camera The camera
    *   @param[in] jitter If jittering should be used
    *   @param[in/out] seed If jitter = true, this is the seed used for the rng
    *   @return A ray from the eye position through the corresponding pixel
    */
    __device__
    Ray
    launchRay(const uint32_t& tid, const uint32_t& width, const uint32_t& height, const Camera& camera, const bool& jitter = false, uint32_t *seed = nullptr);

    /**
    *   @brief Trace a ray through the scene and gather geometry information
    *   @param[in] scene The scene size
    *   @param[in] ray The ray
    *   @return The local geometry information of the intersection point
    */
    __device__
    LocalGeometry
    traceRay(Scene& scene, const Ray& ray);

    /**
    *   @brief Trace a shadow ray to a light source
    *   @param[in] scene The scene
    *   @param[in] lightDist The distance to the light source
    *   @param[in] ray The ray
    *   @return True if the light source is visible, false if it is occluded
    */
    __device__
    bool
    traceVisibility(Scene& scene, const float& lightDist, const Ray& ray);

    /**
    *   @brief Convert a direction to UV coordinates of an environment map
    *   @param[in] direction The direction
    *   @param[in] width The image width
    *   @param[in] height The image height
    *   @return The pixel of the corresponding direction 
    */
    __device__
    Vector2uint32_t
    direction2UV(const Vector3float& direction, const uint32_t& width, const uint32_t& height);
}

#include "../../src/Core/TracingDetail.cuh"

#endif