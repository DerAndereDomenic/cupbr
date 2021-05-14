#ifndef __CUPBR_RENDERER_GRADIENTDOMAIN_CUH
#define __CUPBR_RENDERER_GRADIENTDOMAIN_CUH

#include <Scene/Scene.cuh>
#include <DataStructure/Image.cuh>
#include <DataStructure/Camera.cuh>

namespace PBRendering
{
    /**
    *   @brief An implementation of a gradient domain path tracer
    *   @param[in] scene The scene to render
    *   @param[in] frameIndex The frame index
    *   @param[in] camera The camera
    *   @param[in] maxTraceDepth The maximum number of recursive rays
    *   @param[out] base The base radiance
    *   @param[out] temp A temporary buffer
    *   @param[out] gradient_x The x gradient image 
    *   @param[out] gradient_y The y gradient image
    *   @param[out] output_img The rendered HDR image
    */
    void
    gradientdomain(Scene& scene,
                   const Camera& camera,
                   const uint32_t& frameIndex,
                   const uint32_t& maxTraceDepth,
                   Image<Vector3float>* base,
                   Image<Vector3float>* temp,
                   Image<Vector3float>* gradient_x,
                   Image<Vector3float>* gradient_y,
                   Image<Vector3float>* output_img);
}

#endif