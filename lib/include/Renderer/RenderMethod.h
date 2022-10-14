#ifndef __CUPBR_RENDERER_RENDERMETHOD_H
#define __CUPBR_RENDERER_RENDERMETHOD_H

#include <Core/Plugin.h>
#include <Scene/Scene.h>
#include <DataStructure/Image.h>
#include <DataStructure/Camera.h>

namespace cupbr
{
    class RenderMethod : public Plugin
    {
        public:
        RenderMethod() = default;

        RenderMethod(Properties* properties) {}

        virtual void render(Scene* scene,
                            const Camera& camera,
                            const uint32_t& frameIndex,
                            Image<Vector3float>* output_img) {}
    };
} //namespace cupbr

#endif