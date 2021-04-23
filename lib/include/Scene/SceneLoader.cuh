#ifndef __CUPBR_SCENE_SCENELOADER_CUH
#define __CUPBR_SCENE_SCENELOADER_CUH

#include <Scene/Scene.cuh>

namespace SceneLoader
{
    /**
    *   @brief Creates a cornell box with three spheres
    *   @return A scene pointer 
    */
    Scene
    cornellBoxSphere();

    /**
    *   @brief Destroys the cornell box sphere scene
    *   @param[in] scene The scene to be destroyed
    */
    void
    destroyCornellBoxSphere(Scene scene);
}

#endif