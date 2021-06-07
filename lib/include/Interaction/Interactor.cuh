#ifndef __CUPBR_INTERACTION_INTERACTOR_CUH
#define __CUPBR_INTERACTION_INTERACTOR_CUH

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <memory>

#include <DataStructure/Camera.cuh>
#include <Scene/Scene.cuh>

/**
*   @brief A class to handle user interactions 
*/
class Interactor
{
    public:
        /**
        *   @brief Default constructor 
        */
        Interactor();

        /**
        *   @brief Default destructor 
        */
        ~Interactor();

        /**
        *   @brief Add the window for which the input should be handled 
        *   @param[in] window The window
        */
        void
        registerWindow(GLFWwindow* window);

        /**
        *   @brief Add the scene we want to interact with 
        *   @param[in] scene The scene
        */
        void
        registerScene(Scene& scene);

        /** 
        *   @brief Add the camera
        *   @param[in] camera The camera
        */
        void
        registerCamera(const Camera& camera);

        /**
        *   @brief This handles the user interaction. It should be called every frame
        */
        void
        handleInteraction();
    private:
        class Impl;
        std::unique_ptr<Impl> impl;     /**< Implementation pointer */
};

#endif