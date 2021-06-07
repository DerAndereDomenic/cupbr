#include <Interaction/Interactor.cuh>
#include <Interaction/MousePicker.cuh>
#include <Core/Memory.cuh>
#include <imgui.h>

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

        RenderingMethod method = PATHTRACER;
        ToneMappingType tonemapping = REINHARD;

        bool window_registered = false;
        bool camera_registered = false;
        bool scene_registered = false;
        bool enable_render_settings = false;
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
    if(state == GLFW_PRESS && !(impl->pressed) && !(impl->enable_render_settings))
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
        }
    }

    if(glfwGetKey(impl->window, GLFW_KEY_M) == GLFW_PRESS && !(impl->pressed))
    {
        impl->pressed = true;
        impl->enable_render_settings = !(impl->enable_render_settings);
    }

    if(state == GLFW_RELEASE && glfwGetKey(impl->window, GLFW_KEY_M) == GLFW_RELEASE && impl->pressed)
    {
        impl->pressed = false;
    }

    if(impl->enable_render_settings)
    {
        ImGui::Begin("Render settings", &(impl->enable_render_settings));

        if(ImGui::BeginMenu("Renderer:"))
        {
            if(ImGui::MenuItem("Path Tracing"))
            {
                impl->method = PATHTRACER;
            }
            else if(ImGui::MenuItem("Ray Tracing"))
            {
                impl->method = RAYTRACER;
            }
                
            ImGui::EndMenu();
        }

        if(ImGui::BeginMenu("Tone Mapping:"))
        {
            if(ImGui::MenuItem("Reinhard"))
            {
                impl->tonemapping = REINHARD;
            }
            else if(ImGui::MenuItem("Gamma"))
            {
                impl->tonemapping = GAMMA;
            }
                
            ImGui::EndMenu();
        }

        ImGui::Separator();
        ImGui::Text("Material:");
        if(ImGui::BeginMenu("Type"))
        {
            if(ImGui::MenuItem("LAMBERT"))
            {

            }
            else if(ImGui::MenuItem("PHONG"))
            {

            }
            else if(ImGui::MenuItem("GLASS"))
            {

            }
            else if(ImGui::MenuItem("MIRROR"))
            {

            }
            else if(ImGui::MenuItem("GGX"))
            {

            }
            ImGui::EndMenu();
        }

        float f = impl->host_material->shininess;
        if(impl->host_material->type != GGX)
        {
            f /= 128.0f;
        }

        if(ImGui::SliderFloat("Roughness", &f, 0.0f, 1.0f))
        {
            
        }

        Vector3float d = impl->host_material->albedo_d;
        Vector3float s = impl->host_material->albedo_s;

        float diffuse[] = {d.x, d.y, d.z};
        float specular[] = {s.x, s.y, s.z};

        if(ImGui::ColorEdit3("Albedo diffuse", diffuse))
        {
            
        }
        
        if(ImGui::ColorEdit3("Albedo specular", specular))
        {

        }

        ImGui::End();
    }
    
}

ToneMappingType
Interactor::getToneMapping()
{
    return impl->tonemapping;
}

RenderingMethod
Interactor::getRenderingMethod()
{
    return impl->method;
}