#ifndef __CUPBR_SCENE_SCENELOADER_CUH
#define __CUPBR_SCENE_SCENELOADER_CUH

#include <Scene/Scene.cuh>

namespace cupbr
{
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
        *   @brief Destroys a scene
        *   @param[in] scene The scene to be destroyed
        */
        void
            destroyScene(Scene scene);
    } //namespace SceneLoader
} //namespace cupbr

#endif