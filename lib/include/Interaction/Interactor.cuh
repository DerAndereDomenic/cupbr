#ifndef __CUPBR_INTERACTION_INTERACTOR_CUH
#define __CUPBR_INTERACTION_INTERACTOR_CUH

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <memory>

#include <DataStructure/Camera.cuh>
#include <Scene/Scene.cuh>

class Interactor
{
    public:
        Interactor();

        ~Interactor();

        void
        registerWindow(GLFWwindow* window);

        void
        registerScene(Scene& scene);

        void
        registerCamera(const Camera& camera);

        void
        handleInteraction();
    private:
        class Impl;
        std::unique_ptr<Impl> impl;     /**< Implementation pointer */
};

#endif