#ifndef __CUPBR_RENDERER_RAYTRACER_CUH
#define __CUPBR_RENDERER_RAYTRACER_CUH

#include <Scene/Scene.cuh>
#include <DataStructure/Image.cuh>
#include <DataStructure/Camera.cuh>

namespace PBRendering
{
    /**
    *   @brief Render the scene using conventional ray tracing
    *   @param[in] scene The scene to render
    *   @param[in] scene_size The scene size
    *   @param[out] output_img The resulting HDR rendering 
    */
    void
    raytracing(const Scene scene,
               const uint32_t& scene_size,
               const Camera& camera,
               Image<Vector3float>* output_img);
}

#endif