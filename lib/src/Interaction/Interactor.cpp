#include <Interaction/Interactor.h>
#include <Interaction/MousePicker.h>
#include <Core/Memory.h>
#include <imgui.h>
#include <Core/KeyEvent.h>
#include <Core/MouseEvent.h>
#include <Core/WindowEvent.h>

//#include <filesystem>
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem>

namespace cupbr
{
    class Interactor::Impl
    {
        public:
        Impl();

        ~Impl();

        Window* window;
        int32_t width; // Width of the framebuffer without menu
        int32_t height;
        int32_t menu_width;

        Scene* scene;
        Camera* camera;

        Material* device_material;
        Material* host_material;
        int32_t* scene_index;

        RenderingMethod method = RenderingMethod::PATHTRACER;
        ToneMappingType tonemapping = ToneMappingType::REINHARD;

        //Helper
        bool window_registered = false;
        bool camera_registered = false;
        bool scene_registered = false;
        bool material_update = false;
        bool post_processing = false;
        bool close = false;
        bool edit_mode = true;
        bool reset_scene = false;
        bool use_russian_roulette = false;

        std::string scene_path;

        //Tone Mapping
        float exposure = 1.0f;

        //Bloom
        void compute_threshold();

        float knee = 1.0f;
        float threshold = 1.0f;

        Vector4float threshold_curve;
    };

    Interactor::Impl::Impl()
    {
        device_material = Memory::createDeviceObject<Material>();
        host_material = Memory::createHostObject<Material>();
        scene_index = Memory::createDeviceObject<int32_t>();

        compute_threshold();
    }

    Interactor::Impl::~Impl()
    {
        Memory::destroyDeviceObject<Material>(device_material);
        Memory::destroyHostArray<Material>(host_material);
        Memory::destroyDeviceObject<int32_t>(scene_index);
    }

    Interactor::~Interactor() = default;

    Interactor::Interactor(const RenderingMethod& method)
    {
        impl = std::make_unique<Impl>();
        impl->method = method;
    }

    void
    Interactor::registerWindow(Window* window, const int32_t& menu_width)
    {
        impl->window = window;
        impl->width = window->width() - menu_width;
        impl->height = window->height();
        impl->menu_width = menu_width;

        impl->window_registered = true;
    }

    void
    Interactor::registerScene(Scene* scene)
    {
        impl->scene = scene;

        impl->scene_registered = true;
    }

    void
    Interactor::registerCamera(Camera* camera)
    {
        impl->camera = camera;

        impl->camera_registered = true;
    }

    bool
    Interactor::onEvent(Event& event)
    {
        if (event.getEventType() == EventType::MouseButtonPressed)
        {
            double xpos, ypos;
            glfwGetCursorPos((GLFWwindow*)impl->window->getInternalWindow(), &xpos, &ypos);

            int32_t x = static_cast<int32_t>(xpos);
            int32_t y = impl->width - static_cast<int32_t>(ypos);   //glfw coordinates are flipped

            if (x >= 0 && x < impl->width && y >= 0 && y < impl->height)
            {
                Interaction::pickMouse(x,
                                       y,
                                       impl->width,
                                       impl->height,
                                       *(impl->scene),
                                       *(impl->camera),
                                       impl->device_material,
                                       impl->scene_index);

                Memory::copyDevice2HostObject(impl->device_material, impl->host_material);
                return true;
            }
        }

        if (event.getEventType() == EventType::KeyPressed)
        {
            KeyPressedEvent e = *(KeyPressedEvent*)&event;

            if (e.getKeyCode() == GLFW_KEY_LEFT_ALT)
            {
                impl->edit_mode = !impl->edit_mode;
                if (impl->edit_mode)
                {
                    glfwSetInputMode((GLFWwindow*)(impl->window->getInternalWindow()), GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                }
                else
                {
                    glfwSetInputMode((GLFWwindow*)(impl->window->getInternalWindow()), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                }

                return true;
            }

            if (e.getKeyCode() == GLFW_KEY_ESCAPE)
            {
                impl->close = true;
                return true;
            }
        }

        if (event.getEventType() == EventType::FileDropped)
        {
            FileDroppedEvent e = *(FileDroppedEvent*)&event;

            impl->reset_scene = true;
            impl->scene_path = e.getFilePath();
            return true;
        }

        return false;
    }

    void
    Interactor::handleInteraction()
    {
        if (!impl->window_registered)
        {
            std::cerr << "[Interactor] ERROR: No window registered! Call registerWindow()" << std::endl;
            return;
        }

        if (!impl->camera_registered)
        {
            std::cerr << "[Interactor] ERROR: No camera registered! Call registerCamera()" << std::endl;
            return;
        }

        if (!impl->scene_registered)
        {
            std::cerr << "[Interactor] ERROR: No scene registered! Call registerScene()" << std::endl;
            return;
        }

        if (!impl->edit_mode)
            impl->camera->processInput((GLFWwindow*)(impl->window->getInternalWindow()), impl->window->delta_time());

        impl->material_update = false;

        bool dummy = true;
        if (impl->edit_mode)
        {
            ImGui::SetNextWindowPos(ImVec2(impl->width, 0));
            ImGui::SetNextWindowSize(ImVec2(impl->menu_width, impl->height));
            ImGui::Begin("Render settings", &dummy, ImGuiWindowFlags_MenuBar);

            if (ImGui::BeginMenuBar())
            {
                if (ImGui::BeginMenu("Scene"))
                {
                    std::string path = "res/Scenes";
                    std::vector<std::string> paths;
                    for (auto entry : std::experimental::filesystem::directory_iterator(path))
                    {
                        paths.push_back(entry.path().string());
                    }

                    for (std::string s : paths)
                    {
                        if (ImGui::MenuItem(s.c_str()))
                        {
                            impl->reset_scene = true;
                            impl->scene_path = s;
                        }
                    }

                    ImGui::EndMenu();
                }
                ImGui::EndMenuBar();
            }

            ImGui::Text("Renderer:");
            ImGui::Separator();
            if (ImGui::MenuItem("Path Tracing"))
            {
                impl->method = RenderingMethod::PATHTRACER;
            }
            else if (ImGui::MenuItem("Ray Tracing"))
            {
                impl->method = RenderingMethod::RAYTRACER;
            }
            else if (ImGui::MenuItem("Volume Rendering"))
            {
                impl->method = RenderingMethod::VOLUME;
            }

            ImGui::Checkbox("Russian Roulette", &(impl->use_russian_roulette));

            ImGui::Text("Tone Mapping:");
            ImGui::Separator();
            if (ImGui::MenuItem("Reinhard"))
            {
                impl->tonemapping = ToneMappingType::REINHARD;
            }
            else if (ImGui::MenuItem("Gamma"))
            {
                impl->tonemapping = ToneMappingType::GAMMA;
            }


            ImGui::SliderFloat("Exposure", &(impl->exposure), 0.01f, 10.0f);

            ImGui::Separator();

            ImGui::Text("Material:");
            ImGui::Separator();
            ImGui::Text("Type:");
            if (ImGui::MenuItem("LAMBERT"))
            {
                impl->host_material->type = MaterialType::LAMBERT;
                impl->material_update = true;
            }
            else if (ImGui::MenuItem("PHONG"))
            {
                impl->host_material->type = MaterialType::PHONG;
                impl->material_update = true;
            }
            else if (ImGui::MenuItem("GLASS"))
            {
                impl->host_material->type = MaterialType::GLASS;
                impl->material_update = true;
            }
            else if (ImGui::MenuItem("MIRROR"))
            {
                impl->host_material->type = MaterialType::MIRROR;
                impl->material_update = true;
            }
            else if (ImGui::MenuItem("GGX"))
            {
                impl->host_material->type = MaterialType::GGX;
                impl->material_update = true;
            }
            else if(ImGui::MenuItem("VOLUME"))
            {
                impl->host_material->type = MaterialType::VOLUME;
                impl->material_update = true;
            }

            if (ImGui::SliderFloat("Shininess", &(impl->host_material->shininess), 0.0f, 128.0f))
            {
                impl->material_update = true;
            }

            if (ImGui::SliderFloat("Roughness", &(impl->host_material->roughness), 1e-3f, 1.0f))
            {
                impl->material_update = true;
            }

            if (ImGui::SliderFloat("Eta", &(impl->host_material->eta), 0.0f, 5.0f))
            {
                impl->material_update = true;
            }

            if (ImGui::ColorEdit3("Albedo emissive", reinterpret_cast<float*>(&(impl->host_material->albedo_e))))
            {
                impl->material_update = true;
            }

            if (ImGui::ColorEdit3("Albedo diffuse", reinterpret_cast<float*>(&(impl->host_material->albedo_d))))
            {
                impl->material_update = true;
            }

            if (ImGui::ColorEdit3("Albedo specular", reinterpret_cast<float*>(&(impl->host_material->albedo_s))))
            {
                impl->material_update = true;
            }

            if(ImGui::InputFloat3("Absorption", reinterpret_cast<float*>(&(impl->host_material->volume.sigma_a))))
            {
                impl->material_update = true;
            }

            if(ImGui::InputFloat3("Scattering", reinterpret_cast<float*>(&(impl->host_material->volume.sigma_s))))
            {
                impl->material_update = true;
            }

            if(ImGui::InputFloat("Phase", &(impl->host_material->volume.g)))
            {
                impl->material_update = true;
            }

            if(ImGui::Checkbox("Glass Interface", reinterpret_cast<bool*>(&(impl->host_material->volume.interface))))
            {
                impl->material_update = true;
            }

            ImGui::Separator();
            ImGui::Text("Volume:");

            if (ImGui::InputFloat3("Sigma_a", reinterpret_cast<float*>(&(impl->scene->volume.sigma_a))))
            {
                impl->material_update = true;
            }

            if (ImGui::InputFloat3("Sigma_s", reinterpret_cast<float*>(&(impl->scene->volume.sigma_s))))
            {
                impl->material_update = true;
            }

            if (ImGui::InputFloat("g", &(impl->scene->volume.g)))
            {
                impl->material_update = true;
            }

            ImGui::Separator();
            ImGui::Text("PostProcessing");

            ImGui::Checkbox("Bloom", &(impl->post_processing));

            if (ImGui::SliderFloat("Threshold", &(impl->threshold), 0.0f, 2.0f))
            {
                impl->compute_threshold();
            }

            if (ImGui::SliderFloat("Knee", &(impl->knee), 0.0f, 1.0f))
            {
                impl->compute_threshold();
            }

            ImGui::End();

            if (impl->material_update)
            {
                Memory::copyHost2DeviceObject(impl->host_material, impl->device_material);
                Interaction::updateMaterial(*(impl->scene), impl->scene_index, impl->device_material);
            }
        }
    }

    bool
    Interactor::updated()
    {
        return impl->material_update;
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

    float
    Interactor::getExposure()
    {
        return impl->exposure;
    }

    bool
    Interactor::usePostProcessing()
    {
        return impl->post_processing;
    }

    void
    Interactor::Impl::compute_threshold()
    {
        threshold_curve = Vector4float(threshold, knee - threshold, 2.0f * knee, 0.25f / knee);
    }

    Vector4float
    Interactor::getThreshold()
    {
        return impl->threshold_curve;
    }

    bool
    Interactor::shouldClose()
    {
        return impl->close;
    }

    bool
    Interactor::resetScene(std::string& file_path)
    {
        bool should_reset = impl->reset_scene;
        impl->reset_scene = false;
        file_path = impl->scene_path;
        return should_reset;
    }

    bool
    Interactor::useRussianRoulette()
    {
        return impl->use_russian_roulette;
    }

} //namespace cupbr