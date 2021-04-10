#ifndef __CUPBR_SCENE_SCENELOADER_CUH
#define __CUPBR_SCENE_SCENELOADER_CUH

#include <Scene/Scene.cuh>

namespace SceneLoader
{
    /**
    *   @brief Creates a cornell box with three spheres
    *   @param[out] scene_size The number of objects in the scene
    *   @return A scene pointer 
    */
    Scene
    cornellBoxSphere(uint32_t* scene_size);

    /**
    *   @brief Destroys the cornell box sphere scene
    *   @param[in] scene The scene to be destroyed
    */
    void
    destroyCornellBoxSphere(Scene scene);
}

#endif