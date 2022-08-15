#ifndef __CUPBR_RENDERER_VOLUMERENDERER_H
#define __CUPBR_RENDERER_VOLUMERENDERER_H

#include <Renderer/RenderMethod.h>

namespace cupbr
{
    class VolumeRenderer : public RenderMethod
    {
        public:
        VolumeRenderer() = default;

        VolumeRenderer(Properties* properties) {}

        /**
        *   @brief An implementation of a volume renderer (prototype)
        *   @param[in] scene The scene to render
        *   @param[in] frameIndex The frame index
        *   @param[in] camera The camera
        *   @param[in] maxTraceDepth The maximum number of recursive rays
        *   @param[out] output_img The rendered HDR image
        */
        virtual void render(Scene& scene,
                            const Camera& camera,
                            const uint32_t& frameIndex,
                            const uint32_t& maxTraceDepth,
                            const bool& useRussianRoulette,
                            Image<Vector3float>* output_img);
    };
} //namespace cupbr

#endif