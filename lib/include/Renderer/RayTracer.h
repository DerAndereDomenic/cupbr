#ifndef __CUPBR_RENDERER_RAYTRACER_H
#define __CUPBR_RENDERER_RAYTRACER_H

#include <Scene/Scene.h>
#include <DataStructure/Image.h>
#include <DataStructure/Camera.h>

namespace cupbr
{
    namespace PBRendering
    {
        /**
        *   @brief Render the scene using conventional ray tracing
        *   @param[in] scene The scene to render
        *   @param[out] output_img The resulting HDR rendering
        */
        void raytracing(Scene& scene,
                        const Camera& camera,
                        Image<Vector3float>* output_img);
    } //namespace PBRendering
} //namespace cupbr

#endif