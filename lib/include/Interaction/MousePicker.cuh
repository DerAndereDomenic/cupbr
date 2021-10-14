#ifndef __CUPBR_INTERACTION_MOUSEPICKER_CUH
#define __CUPBR_INTERACTION_MOUSEPICKER_CUH

#include <cstdint>
#include <Scene/Scene.cuh>
#include <DataStructure/Camera.cuh>
#include <Geometry/Material.cuh>

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
        *   @param[out] outMaterial The material found at the selected pixel
        *   @param[out] sceneIndex The scene index of the hit object
        */
        void pickMouse(const uint32_t& x,
                       const uint32_t& y,
                       const uint32_t& width,
                       const uint32_t& height,
                       Scene& scene,
                       Camera& camera,
                       Material* outMaterial,
                       int32_t* sceneIndex);

        /**
        *   @brief Update the material at the object defined by scene_index
        *   @param[in] scene The scene
        *   @param[in] scene_index The index of the object which material we want to change
        *   @param[in] newMaterial The new material
        */
        void updateMaterial(Scene& scene,
                            int32_t* scene_index,
                            Material* newMaterial);
    } //namespace Interaction
} //namespace cupbr

#endif