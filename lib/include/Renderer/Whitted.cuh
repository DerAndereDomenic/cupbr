#ifndef __CUPBR_RENDERER_WHITTED_CUH
#define __CUPBR_RENDERER_WHITTED_CUH

#include <Scene/Scene.cuh>
#include <DataStructure/Image.cuh>
#include <DataStructure/Camera.cuh>

namespace PBRendering
{
    /**
    *   @brief An implementation of a whitted ray tracer
    *   @param[in] scene The scene to render
    *   @param[in] scene_size The scene size
    *   @param[in] camera The camera
    *   @param[in] maxTraceDepth The maximum number of recursive rays
    *   @param[out] output_img The rendered HDR image
    */
    void
    whitted(const Scene scene,
            const uint32_t& scene_size,
            const Camera& camera,
            const uint32_t& maxTraceDepth,
            Image<Vector3float>* output_img);
}

#endif