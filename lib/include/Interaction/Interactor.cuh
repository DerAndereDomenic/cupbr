#ifndef __CUPBR_INTERACTION_INTERACTOR_CUH
#define __CUPBR_INTERACTION_INTERACTOR_CUH

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <memory>

class Interactor
{
    public:
        Interactor();

        void
        registerWindow(GLFWwindow* window);

        void
        updateInteraction();
    private:
        class Impl;
        std::unique_ptr<Impl> impl;     /**< Implementation pointer */
};

#endif