#ifndef __CUPBR_INTERACTION_MOUSEPICKER_CUH
#define __CUPBR_INTERACTION_MOUSEPICKER_CUH

#include <cstdint>
#include <Scene/Scene.cuh>
#include <DataStructure/Camera.cuh>
#include <Geometry/Geometry.cuh>

namespace Interaction
{
    void pickMouse(const uint32_t x,
                   const uint32_t y,
                   Scene& scene,
                   Camera& camera,
                   Geometry* outGeometry);
}

#endif