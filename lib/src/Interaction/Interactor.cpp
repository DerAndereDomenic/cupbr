#include <Interaction/Interactor.h>
#include <Interaction/MousePicker.h>
#include <Core/Memory.h>
#include <imgui.h>
#include <Core/KeyEvent.h>
#include <Core/MouseEvent.h>
#include <Core/WindowEvent.h>
#include <iomanip>

#include <Scene/SceneLoader.h>

#include <filesystem>

namespace cupbr
{
    class Interactor::Impl
    {
        public:
        Impl();

        ~Impl();

        Window* window = nullptr;
        int32_t width; // Width of the framebuffer without menu
        int32_t height;

        Scene* scene = nullptr;
        Camera* camera = nullptr;
        PBRenderer* renderer = nullptr;
        ToneMapper* mapper = nullptr;

        int32_t* dev_scene_index;
        int32_t scene_index = 0;

        //Helper
        bool material_update = false;
        bool post_processing = false;
        bool close = false;
        bool edit_mode = true;
        bool reset_scene = false;
        bool use_russian_roulette = false;

        std::string scene_path;
        std::string fps_string;

        float time = 0;

        //Tone Mapping
        float exposure = 1.0f;

        //Bloom
        void compute_threshold();

        float knee = 1.0f;
        float threshold = 1.0f;

        Vector4float threshold_curve;

        void createMenuFromProperties(const std::string& name, Properties& properties);
    };

    Interactor::Impl::Impl()
    {
        dev_scene_index = Memory::createDeviceObject<int32_t>();

        compute_threshold();
    }

    Interactor::Impl::~Impl()
    {
        Memory::destroyDeviceObject<int32_t>(dev_scene_index);
    }

    Interactor::~Interactor() = default;

    Interactor::Interactor()
    {
        impl = std::make_unique<Impl>();
    }

    void
    Interactor::registerWindow(Window* window)
    {
        impl->window = window;
        impl->width = window->width();
        impl->height = window->height();
    }

    void
    Interactor::registerScene(Scene* scene)
    {
        impl->scene = scene;
    }

    void
    Interactor::registerCamera(Camera* camera)
    {
        impl->camera = camera;
    }

    void 
    Interactor::registerRenderer(PBRenderer* renderer)
    {
        impl->renderer = renderer;
    }

    void 
    Interactor::registerToneMapper(ToneMapper* mapper)
    {
        impl->mapper = mapper;
    }

    bool
    Interactor::onEvent(Event& event)
    {
        if (event.getEventType() == EventType::MouseButtonPressed)
        {
            Vector2float mouse_pos = impl->window->getMousePosition();

            Vector2float view_port_size = impl->window->getViewportSize();
            int32_t x = static_cast<int32_t>(mouse_pos.x);
            int32_t y = view_port_size.y - static_cast<int32_t>(mouse_pos.y);   //glfw coordinates are flipped

            if (x >= 0 && x < view_port_size.x && y >= 0 && y < view_port_size.y)
            {
                Interaction::pickMouse(x,
                                       y,
                                       view_port_size.x,
                                       view_port_size.y,
                                       *(impl->scene),
                                       *(impl->camera),
                                       impl->dev_scene_index);

                Memory::copyDevice2HostObject<int32_t>(impl->dev_scene_index, &(impl->scene_index));

                return true;
            }
        }

        if (event.getEventType() == EventType::KeyPressed)
        {
            KeyPressedEvent e = *(KeyPressedEvent*)&event;

            if (e.getKeyCode() == Key::LeftAlt)
            {
                impl->edit_mode = !impl->edit_mode;
                impl->window->setEditMode(impl->edit_mode);
                impl->camera->stop(impl->window);

                return true;
            }

            if (e.getKeyCode() == Key::Escape)
            {
                impl->close = true;
                return true;
            }
        }

        if (event.getEventType() == EventType::FileDropped)
        {
            FileDroppedEvent e = *(FileDroppedEvent*)&event;

            impl->reset_scene = true;
            Vector2float viewport_size = impl->window->getViewportSize();
            impl->camera->onResize(viewport_size.x / viewport_size.y);
            impl->scene_path = e.getFilePath();
            return true;
        }

        if(event.getEventType() == EventType::WindowResized)
        {
            WindowResizedEvent e = *(WindowResizedEvent*)&event;

            uint32_t width = e.width();
            uint32_t height = e.height();
            impl->camera->onResize(static_cast<float>(width) / static_cast<float>(height));
            impl->window->onResize(width, height);
            impl->renderer->setOutputSize(width, height);
            impl->mapper->registerImage(impl->renderer->getOutputImage());
            impl->renderer->reset();
        }

        return false;
    }

    void 
    Interactor::Impl::createMenuFromProperties(const std::string& name, Properties& properties)
    {
        ImGui::Begin((name + " Editor").c_str(), nullptr, ImGuiWindowFlags_MenuBar);

        if (ImGui::BeginMenuBar())
        {
            if (ImGui::BeginMenu((name + " Name").c_str()))
            {
                for (auto it = PluginManager::begin(); it != PluginManager::end(); ++it)
                {
                    if (it->second->get_super_name() == name && ImGui::MenuItem((it->first).c_str()))
                    {
                        properties.reset();
                        properties.setProperty("name", it->first);
                        material_update = true;
                    }
                }
                ImGui::EndMenu();
            }
            ImGui::EndMenuBar();
        }

        for(auto it = properties.begin(); it != properties.end(); ++it)
        {
            if(std::holds_alternative<std::string>(it->second))
            {
                ImGui::Text(std::get<std::string>(it->second).c_str());
            }
            else if(std::holds_alternative<float>(it->second))
            {
                float val = std::get<float>(it->second);
                if(ImGui::InputFloat((it->first).c_str(), &val))
                {
                    material_update = true;
                    properties.setProperty(it->first, val);
                }
            }
            else if(std::holds_alternative<Vector3float>(it->second))
            {
                float* val = reinterpret_cast<float*>(&std::get<Vector3float>(it->second));
                if(ImGui::ColorEdit3((it->first).c_str(), val))
                {
                    material_update = true;
                    properties.setProperty(it->first, Vector3float(val[0], val[1], val[2]));
                }
            }
        }

        ImGui::End();
    }

    void
    Interactor::handleInteraction()
    {
        if (!impl->window)
        {
            std::cerr << "[Interactor] ERROR: No window registered! Call registerWindow()" << std::endl;
            return;
        }

        if (!impl->camera)
        {
            std::cerr << "[Interactor] ERROR: No camera registered! Call registerCamera()" << std::endl;
            return;
        }

        if (!impl->scene)
        {
            std::cerr << "[Interactor] ERROR: No scene registered! Call registerScene()" << std::endl;
            return;
        }

        if (!impl->edit_mode)
            impl->camera->processInput(impl->window, impl->window->delta_time());

        impl->material_update = false;

        if (!impl->edit_mode)
            ImGui::BeginDisabled();

        ImGui::Begin("Statistics", nullptr, ImGuiWindowFlags_MenuBar);

        if (ImGui::BeginMenuBar())
        {
            if (ImGui::BeginMenu("Scene"))
            {
                std::string path = "res/Scenes";
                std::vector<std::string> paths;
                for (auto entry : std::filesystem::directory_iterator(path))
                {
                    paths.push_back(entry.path().string());
                }

                for (std::string s : paths)
                {
                    if (ImGui::MenuItem(s.c_str()))
                    {
                        impl->reset_scene = true;
                        impl->scene_path = s;
                        impl->scene_index = 0;
                        Vector2float viewport_size = impl->window->getViewportSize();
                        impl->camera->onResize(viewport_size.x / viewport_size.y);
                    }
                }

                ImGui::EndMenu();
            }
            ImGui::EndMenuBar();
        }

        if (impl->time >= 1000)
        {
            std::stringstream render_time_stream, fps_stream;
            render_time_stream << std::fixed << std::setprecision(2) << impl->window->delta_time() * 1000.0f;
            fps_stream << std::fixed << std::setprecision(2) << 1.0f / impl->window->delta_time();
            impl->fps_string = ("Render time: " + render_time_stream.str() + "ms : " + fps_stream.str() + "fps");
            impl->time = 0;
        }

        impl->time += impl->window->delta_time() * 1000.0f;

        ImGui::Text(impl->fps_string.c_str());

        if (ImGui::BeginMenuBar())
        {
            if (ImGui::BeginMenu("Renderer"))
            {
                for(auto it = PluginManager::begin(); it != PluginManager::end(); ++it)
                {
                    if (it->second->get_super_name() == "RenderMethod" && ImGui::MenuItem((it->first).c_str()))
                    {
                        impl->material_update = true;
                        impl->renderer->changeRenderMethod(it->first);
                    }
                }
                ImGui::EndMenu();
            }
            ImGui::EndMenuBar();
        }

        ImGui::Text("Tone Mapping:");
        ImGui::Separator();
        if (ImGui::MenuItem("Reinhard"))
        {
            if (impl->mapper)
                impl->mapper->setType(ToneMappingType::REINHARD);
        }
        else if (ImGui::MenuItem("Gamma"))
        {
            if (impl->mapper)
                impl->mapper->setType(ToneMappingType::GAMMA);
        }


        if(ImGui::SliderFloat("Exposure", &(impl->exposure), 0.01f, 10.0f))
        {
            impl->mapper->setExposure(impl->exposure);
        }

        if(ImGui::Button("Screenshot"))
        {
            auto t = std::time(nullptr);
            auto tm = *std::localtime(&t);
            std::ostringstream oss;
            oss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
            impl->mapper->saveToFile("bin/" + oss.str() + ".png");
        }

        ImGui::End();
        
        Properties& properties = impl->scene->properties[impl->scene_index];

        impl->createMenuFromProperties("Material", properties);

        if (impl->material_update)
        {
            SceneLoader::reinitializeScene(impl->scene);
            impl->renderer->reset();
        }
        
        if (!impl->edit_mode)
            ImGui::EndDisabled();
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
    Interactor::usePostProcessing()
    {
        return impl->post_processing;
    }

    bool
    Interactor::resetScene(std::string& file_path)
    {
        bool should_reset = impl->reset_scene;
        impl->reset_scene = false;
        if (should_reset)
            file_path = impl->scene_path;
        return should_reset;
    }

} //namespace cupbr