#include <Interaction/Interactor.cuh>
#include <Interaction/MousePicker.cuh>
#include <Core/Memory.cuh>

class Interactor::Impl
{
    public:
        Impl();
        
        ~Impl();

        GLFWwindow* window;
        int32_t width;
        int32_t height;

        Scene scene;
        Camera camera;

        Material* device_material;
        Material* host_material;

        bool window_registered = false;
        bool camera_registered = false;
        bool scene_registered = false;
        bool pressed = false;
};

Interactor::Impl::Impl()
{
    device_material = Memory::allocator()->createDeviceObject<Material>();
    host_material = Memory::allocator()->createHostObject<Material>();
}

Interactor::Impl::~Impl()
{
    Memory::allocator()->destroyDeviceObject<Material>(device_material);
    Memory::allocator()->destroyHostArray<Material>(host_material);
}

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
        int32_t y = impl->width - static_cast<int32_t>(ypos);   //glfw coordinates are flipped

        if(x >= 0 && x < impl->width && y >= 0 && y < impl->height)
        {
            Interaction::pickMouse(x,
                                   y,
                                   impl->width,
                                   impl->height,
                                   impl->scene,
                                   impl->camera,
                                   impl->device_material);

            Memory::allocator()->copyDevice2HostObject(impl->device_material, impl->host_material);

            //TODO: Make material changable
            printf("Type: %i\n", impl->host_material->type);
            printf("Diffuse: %f %f %f\n", impl->host_material->albedo_d.x,
                                          impl->host_material->albedo_d.y,
                                          impl->host_material->albedo_d.z);
            printf("Specular: %f %f %f\n", impl->host_material->albedo_s.x,
                                           impl->host_material->albedo_s.y,
                                           impl->host_material->albedo_s.z);
        }
    }

    if(state == GLFW_RELEASE && impl->pressed)
    {
        impl->pressed = false;
    }
    
}