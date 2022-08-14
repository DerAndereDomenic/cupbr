#ifndef __CUPBR_SCENE_SCENELOADER_H
#define __CUPBR_SCENE_SCENELOADER_H

#include <Scene/Scene.h>

namespace cupbr
{
    namespace SceneLoader
    {
        /**
        *   @brief Loads a scene from file
        *   @param[in] path The path to the scene file
        *   @return The scene
        */
        Scene loadFromFile(const std::string& path);

        /**
        *   @brief Reinitializes the scene.
        *   When the properties of the scene were changed, new materials have to be generated.
        *   This is done by this method. The geometry stays unchanged.
        *   @param scene The scene to update
        */
        void reinitializeScene(Scene* scene);

        /**
        *   @brief Destroys a scene
        *   @param[in] scene The scene to be destroyed
        */
        void destroyScene(Scene& scene);
    } //namespace SceneLoader
} //namespace cupbr

#endif