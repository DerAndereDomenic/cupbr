#ifndef __CUPBR_INTERACTION_MOUSEPICKER_H
#define __CUPBR_INTERACTION_MOUSEPICKER_H

#include <cstdint>
#include <Scene/Scene.h>
#include <DataStructure/Camera.h>
#include <Geometry/Material.h>

namespace cupbr
{
    namespace Interaction
    {
        /**
        *   @brief Reads the material of the currently selected pixel
        *   @param[in] x x-position of the mouse
        *   @param[in] y y-position of the mouse
        *   @param[in] width The window width
        *   @param[in] height The window height
        *   @param[in] scene The Scene
        *   @param[in] camera The camera
        *   @param[out] sceneIndex The scene index of the hit object
        */
        void pickMouse(const uint32_t& x,
                       const uint32_t& y,
                       const uint32_t& width,
                       const uint32_t& height,
                       Scene* scene,
                       Camera& camera,
                       int32_t* sceneIndex);
    } //namespace Interaction
} //namespace cupbr

#endif