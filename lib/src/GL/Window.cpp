#include <GL/Window.h>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

#include <glad/glad.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <Core/KeyEvent.h>
#include <Core/MouseEvent.h>
#include <Core/WindowEvent.h>

#include <iostream>

#include <filesystem>

namespace cupbr
{
    static bool s_glfw_initialized = false;
    static bool s_glad_initialized = false;
    static bool s_imgui_initialized = false;

    Window::Window(const char* title, const uint32_t& width, const uint32_t& height)
        :_width(width),
        _height(height)
    {
        if (!s_glfw_initialized)
        {
            if (glfwInit())
            {
                s_glfw_initialized = true;
            }
        }

        glfwSetErrorCallback([](int error, const char* message){
            std::cerr << "Error: " << error << " occured: " << message << std::endl;
        });

        _internal_window = glfwCreateWindow(width, height, title, NULL, NULL);

        glfwMakeContextCurrent(_internal_window);
        glfwSwapInterval(0);

        if (!s_glad_initialized)
        {
            if (gladLoadGL())
            {
                s_glad_initialized = true;
            }
        }

        glfwSetWindowUserPointer(_internal_window, &_event_callback);

        glfwSetMouseButtonCallback(_internal_window, [](GLFWwindow* window, int button, int action, int mode)
        {

            EventCallbackFn fnc = *(EventCallbackFn*)glfwGetWindowUserPointer(window);

            switch (action)
            {
                case GLFW_PRESS:
                {
                    MouseButtonPressedEvent event = MouseButtonPressedEvent(button);
                    fnc(event);
                    break;
                }
                case GLFW_RELEASE:
                {
                    MouseButtonReleasedEvent event = MouseButtonReleasedEvent(button);
                    fnc(event);
                    break;
                }
            }
        });

        glfwSetCursorPosCallback(_internal_window, [](GLFWwindow* window, double x, double y)
        {

            EventCallbackFn fnc = *(EventCallbackFn*)glfwGetWindowUserPointer(window);

            MouseMovedEvent event = MouseMovedEvent(float(x), float(y));
            fnc(event);
        });

        glfwSetKeyCallback(_internal_window, [](GLFWwindow* window, int key, int scancode, int action, int mods)
        {

            EventCallbackFn fnc = *(EventCallbackFn*)glfwGetWindowUserPointer(window);

            switch (action)
            {
                case GLFW_PRESS:
                {
                    KeyPressedEvent event = KeyPressedEvent(key, 0);
                    fnc(event);
                    break;
                }
                case GLFW_RELEASE:
                {
                    KeyReleasedEvent event = KeyReleasedEvent(key);
                    fnc(event);
                    break;
                }
                case GLFW_REPEAT:
                {
                    KeyPressedEvent event = KeyPressedEvent(key, 1);
                    fnc(event);
                    break;
                }
            }
        });

        glfwSetDropCallback(_internal_window, [](GLFWwindow* window, int num_paths, const char* paths[])
        {

            EventCallbackFn fnc = *(EventCallbackFn*)glfwGetWindowUserPointer(window);

            FileDroppedEvent event = FileDroppedEvent(paths[0]);

            fnc(event);
        });

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io;

        io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

        io.FontDefault = io.Fonts->AddFontFromFileTTF("res/Fonts/opensans/static/OpenSans/OpenSans-Regular.ttf", 18.0f);

        if(!std::filesystem::exists("imgui.ini"))
            ImGui::LoadIniSettingsFromDisk("Default.ini");

        ImGui::StyleColorsDark();

        ImGuiStyle& style = ImGui::GetStyle();
        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
        {
            style.WindowRounding = 0.0f;
            style.Colors[ImGuiCol_WindowBg].w = 1.0f;
        }

        setDarkThemeColors();

        ImGui_ImplGlfw_InitForOpenGL(_internal_window, true);
        ImGui_ImplOpenGL3_Init("#version 330");

        s_imgui_initialized = true;

        _renderer = std::make_unique<GLRenderer>(width, height);
    }

    Window::~Window()
    {
        if (s_imgui_initialized)
        {
            ImGui_ImplOpenGL3_Shutdown();
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext();

            s_imgui_initialized = false;
        }

        if (s_glfw_initialized)
        {
            glfwTerminate();

            s_glfw_initialized = false;
        }
    }

    void
    Window::imguiBegin()
    {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        //Create the Dock Space window
        ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
        ImGuiViewport* viewport = ImGui::GetMainViewport();
		ImGui::SetNextWindowPos(viewport->Pos);
		ImGui::SetNextWindowSize(viewport->Size);
		ImGui::SetNextWindowViewport(viewport->ID);
		ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
		ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        //Set the window over the whole viewport and make it non-editable
		window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
		window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

        //Begin Dock Space window
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
        ImGui::Begin("DockSpace", nullptr, window_flags);
        ImGui::PopStyleVar(3);

        //Create Dock Space
        ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
        ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f));
    }

    void
    Window::imguiEnd()
    {
        ImGui::End();   //End the window containing the dock space
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        GLFWwindow* backup_current_context = glfwGetCurrentContext();
        ImGui::UpdatePlatformWindows();
        ImGui::RenderPlatformWindowsDefault();
        glfwMakeContextCurrent(backup_current_context);
    }

    void
    Window::spinOnce()
    {
        float current_time = static_cast<float>(glfwGetTime());
        _delta_time = current_time - _last_time;
        _last_time = current_time;
        glfwSwapBuffers(_internal_window);
        glfwPollEvents();
    }

    Vector2float 
    Window::getMousePosition() const
    {
        double xpos, ypos;
        glfwGetCursorPos(_internal_window, &xpos, &ypos);
        return Vector2float(static_cast<float>(xpos), static_cast<float>(ypos));
    }

    void 
    Window::setEditMode(const bool edit_mode)
    {
        edit_mode ? 
            glfwSetInputMode(_internal_window, GLFW_CURSOR, GLFW_CURSOR_NORMAL) :
            glfwSetInputMode(_internal_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    }

    bool 
    Window::shouldClose() const
    {
        return glfwWindowShouldClose(_internal_window);
    }

    void 
    Window::close()
    {
        glfwDestroyWindow(_internal_window);
    }

    void 
    Window::onResize(const float& width, const float& height)
    {
        _width = width;
        _height = height;
        _renderer->setViewport(0, 0, width, height);
        _renderer->onResize(width, height);
    }

    void 
    Window::displayImage(const RenderBuffer& img)
    {
        bool resized = _renderer->displayImage(img);

        if(resized)
        {
            EventCallbackFn fnc = *(EventCallbackFn*)glfwGetWindowUserPointer(_internal_window);

            Vector2float viewport_size = _renderer->getViewportSize();
            WindowResizedEvent event = WindowResizedEvent(static_cast<uint32_t>(viewport_size.x), static_cast<uint32_t>(viewport_size.y));

            fnc(event);
        }
    }

    Vector2float 
    Window::getWindowPosition() const
    {
        int x,y;
        glfwGetWindowPos(_internal_window, &x, &y);
        return Vector2float(x,y);
    }

    void
    Window::setDarkThemeColors()
    {
        //Hazels color theme for now (https://github.com/TheCherno/Hazel)
        auto& colors = ImGui::GetStyle().Colors;
		colors[ImGuiCol_WindowBg] = ImVec4{ 0.1f, 0.105f, 0.11f, 1.0f };

		// Headers
		colors[ImGuiCol_Header] = ImVec4{ 0.2f, 0.205f, 0.21f, 1.0f };
		colors[ImGuiCol_HeaderHovered] = ImVec4{ 0.3f, 0.305f, 0.31f, 1.0f };
		colors[ImGuiCol_HeaderActive] = ImVec4{ 0.15f, 0.1505f, 0.151f, 1.0f };
		
		// Buttons
		colors[ImGuiCol_Button] = ImVec4{ 0.2f, 0.205f, 0.21f, 1.0f };
		colors[ImGuiCol_ButtonHovered] = ImVec4{ 0.3f, 0.305f, 0.31f, 1.0f };
		colors[ImGuiCol_ButtonActive] = ImVec4{ 0.15f, 0.1505f, 0.151f, 1.0f };

		// Frame BG
		colors[ImGuiCol_FrameBg] = ImVec4{ 0.2f, 0.205f, 0.21f, 1.0f };
		colors[ImGuiCol_FrameBgHovered] = ImVec4{ 0.3f, 0.305f, 0.31f, 1.0f };
		colors[ImGuiCol_FrameBgActive] = ImVec4{ 0.15f, 0.1505f, 0.151f, 1.0f };

		// Tabs
		colors[ImGuiCol_Tab] = ImVec4{ 0.15f, 0.1505f, 0.151f, 1.0f };
		colors[ImGuiCol_TabHovered] = ImVec4{ 0.38f, 0.3805f, 0.381f, 1.0f };
		colors[ImGuiCol_TabActive] = ImVec4{ 0.28f, 0.2805f, 0.281f, 1.0f };
		colors[ImGuiCol_TabUnfocused] = ImVec4{ 0.15f, 0.1505f, 0.151f, 1.0f };
		colors[ImGuiCol_TabUnfocusedActive] = ImVec4{ 0.2f, 0.205f, 0.21f, 1.0f };

		// Title
		colors[ImGuiCol_TitleBg] = ImVec4{ 0.15f, 0.1505f, 0.151f, 1.0f };
		colors[ImGuiCol_TitleBgActive] = ImVec4{ 0.15f, 0.1505f, 0.151f, 1.0f };
		colors[ImGuiCol_TitleBgCollapsed] = ImVec4{ 0.15f, 0.1505f, 0.151f, 1.0f };
    }

} //namespace cupbr
