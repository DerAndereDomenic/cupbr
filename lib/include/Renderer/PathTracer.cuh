#ifndef __CUPBR_RENDERER_PATHTRACER_CUH
#define __CUPBR_RENDERER_PATHTRACER_CUH

#include <Scene/Scene.cuh>
#include <DataStructure/Image.cuh>
#include <DataStructure/Camera.cuh>

namespace cupbr
{
    namespace PBRendering
    {
        /**
        *   @brief An implementation of a path tracer
        *   @param[in] scene The scene to render
        *   @param[in] frameIndex The frame index
        *   @param[in] camera The camera
        *   @param[in] maxTraceDepth The maximum number of recursive rays
        *   @param[in] If russian roulette should be used
        *   @param[out] output_img The rendered HDR image
        */
        void pathtracing(Scene& scene,
                         const Camera& camera,
                         const uint32_t& frameIndex,
                         const uint32_t& maxTraceDepth,
                         const bool& useRussianRoulette,
                         Image<Vector3float>* output_img);
    } //namespace PBRendering
} //namespace cupbr

#endif