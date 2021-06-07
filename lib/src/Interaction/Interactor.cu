#include <Interaction/Interactor.cuh>
#include <Interaction/MousePicker.cuh>

class Interactor::Impl
{
    public:
        Impl() = default;
        
        ~Impl() = default;

        GLFWwindow* window;
        int32_t width;
        int32_t height;

        Scene scene;
        Camera camera;

        bool window_registered = false;
        bool camera_registered = false;
        bool scene_registered = false;
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
Interactor::registerScene(Scene& scene)
{
    impl->scene = scene;

    impl->scene_registered = true;
}

void
Interactor::registerCamera(const Camera& camera)
{
    impl->camera = camera;

    impl->camera_registered = true;
}

void
Interactor::handleInteraction()
{
    if(!impl->window_registered)
    {
        std::cerr << "[Interactor] ERROR: No window registered! Call registerWindow()" << std::endl;
        return;
    }

    if(!impl->camera_registered)
    {
        std::cerr << "[Interactor] ERROR: No camera registered! Call registerCamera()" << std::endl;
        return;
    }

    if(!impl->scene_registered)
    {
        std::cerr << "[Interactor] ERROR: No scene registered! Call registerScene()" << std::endl;
        return;
    }

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