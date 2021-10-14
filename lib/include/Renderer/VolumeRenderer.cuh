#ifndef __CUPBR_RENDERER_VOLUMERENDERER_CUH
#define __CUPBR_RENDERER_VOLUMERENDERER_CUH

#include <Scene/Scene.cuh>
#include <DataStructure/Image.cuh>
#include <DataStructure/Camera.cuh>

namespace cupbr
{
    namespace PBRendering
    {
        /**
        *   @brief An implementation of a volume renderer (prototype)
        *   @param[in] scene The scene to render
        *   @param[in] frameIndex The frame index
        *   @param[in] camera The camera
        *   @param[in] maxTraceDepth The maximum number of recursive rays
        *   @param[out] output_img The rendered HDR image
        */
        void volumetracing(Scene& scene,
                           const Camera& camera,
                           const uint32_t& frameIndex,
                           const uint32_t& maxTraceDepth,
                           Image<Vector3float>* output_img);
    } //namepsace PBRendering
} //namespace cupbr

#endif