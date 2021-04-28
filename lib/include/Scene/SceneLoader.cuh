#ifndef __CUPBR_SCENE_SCENELOADER_CUH
#define __CUPBR_SCENE_SCENELOADER_CUH

#include <Scene/Scene.cuh>

namespace SceneLoader
{
    /**
    *   @brief Loads a scene from file
    *   @param[in] path The path to the scene file
    *   @return The scene 
    */
    Scene
    loadFromFile(const std::string& path);

    /**
    *   @brief Creates a cornell box with three spheres
    *   @return A scene pointer 
    */
    Scene
    cornellBoxSphere();

    /**
    *   @brief Creates a cornell box with three spheres and an area light
    *   @return A scene pointer 
    */
    Scene
    cornellBoxSphereAreaLight();

    /**
    *   @brief Creates a cornell box with three spheres and multiple light sources
    *   @return A scene pointer 
    */
    Scene
    cornellBoxSphereMultiLight();

    /**
    *   @brief Destroys the cornell box sphere scene
    *   @param[in] scene The scene to be destroyed
    */
    void
    destroyCornellBoxSphere(Scene scene);
}

#endif