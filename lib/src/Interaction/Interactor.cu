#include <Interaction/Interactor.cuh>

class Interactor::Impl
{
    public:
        Impl() = default;
        
        ~Impl() = default;

        GLFWwindow* window;
        int32_t width;
        int32_t height;

        bool window_registered = false;
        bool pressed = false;
};

Interactor::~Interactor() = default;

Interactor::Interactor()
{
    impl = std::make_unique<Impl>();
}

void
Interactor::registerWindow(GLFWwindow* window)
{
    impl->window = window;
    glfwGetWindowSize(impl->window, &(impl->width), &(impl->height));

    impl->window_registered = true;
}

void
Interactor::handleInteraction()
{
    int32_t state = glfwGetMouseButton(impl->window, GLFW_MOUSE_BUTTON_LEFT);
    if(state == GLFW_PRESS && !(impl->pressed))
    {
        impl->pressed = true;
        double xpos, ypos;
        glfwGetCursorPos(impl->window, &xpos, &ypos);

        int32_t x = static_cast<int32_t>(xpos);
        int32_t y = static_cast<int32_t>(ypos);

        if(x >= 0 && x < impl->width && y >= 0 && y < impl->height)
        {
            //do mouse picking
        }
    }

    if(state == GLFW_RELEASE && impl->pressed)
    {
        impl->pressed = false;
    }
    
}