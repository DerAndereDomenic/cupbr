#ifndef __CUPBR_RENDERER_WHITTED_CUH
#define __CUPBR_RENDERER_WHITTED_CUH

#include <Scene/Scene.cuh>
#include <DataStructure/Image.cuh>
#include <DataStructure/Camera.cuh>

namespace cupbr 
{
    namespace PBRendering
    {
        /**
        *   @brief An implementation of a whitted ray tracer
        *   @param[in] scene The scene to render
        *   @param[in] camera The camera
        *   @param[in] maxTraceDepth The maximum number of recursive rays
        *   @param[out] output_img The rendered HDR image
        */
        void
            whitted(Scene& scene,
                const Camera& camera,
                const uint32_t& maxTraceDepth,
                Image<Vector3float>* output_img);
    } // namespace PBRendering
} //namespace cupbr

#endif