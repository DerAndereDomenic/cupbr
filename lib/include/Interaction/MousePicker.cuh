#ifndef __CUPBR_INTERACTION_MOUSEPICKER_CUH
#define __CUPBR_INTERACTION_MOUSEPICKER_CUH

#include <cstdint>
#include <Scene/Scene.cuh>
#include <DataStructure/Camera.cuh>
#include <Geometry/Material.cuh>

namespace Interaction
{
    void pickMouse(const uint32_t& x,
                   const uint32_t& y,
                   const uint32_t& width,
                   const uint32_t& height,
                   Scene& scene,
                   Camera& camera,
                   Material* outMaterial);
}

#endif