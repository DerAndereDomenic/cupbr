#include <GL/Window.h>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

#include <glad/glad.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <Core/KeyEvent.h>
#include <Core/MouseEvent.h>

namespace cupbr
{
    static bool s_glfw_initialized = false;
    static bool s_glad_initialized = false;
    static bool s_imgui_initialized = false;

    Window::Window(const char* title, const uint32_t& width, const uint32_t& height)
        :_width(width),
         _height(height)
    {
        if(!s_glfw_initialized)
        {
            if(glfwInit())
            {
                s_glfw_initialized = true;
            }
        }

        _internal_window = glfwCreateWindow(width, height, title, NULL, NULL);

        glfwMakeContextCurrent(_internal_window);
        glfwSwapInterval(0);

        if(!s_glad_initialized)
        {
            if(gladLoadGL())
            {
                s_glad_initialized = true;
            }
        }

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io;

        ImGui::StyleColorsDark();

        ImGui_ImplGlfw_InitForOpenGL(_internal_window, true);
        ImGui_ImplOpenGL3_Init("#version 330");

        s_imgui_initialized = true;

        glfwSetWindowUserPointer(_internal_window, &_event_callback);

        glfwSetMouseButtonCallback(_internal_window, [](GLFWwindow* window, int button, int action, int mode)
        {
            EventCallbackFn fnc = *(EventCallbackFn*)glfwGetWindowUserPointer(window);

            switch(action)
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

            switch(action)
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
    }

    Window::~Window()
    {
        if(s_imgui_initialized)
        {
            ImGui_ImplOpenGL3_Shutdown();
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext();

            s_imgui_initialized = false;
        }

        if(s_glfw_initialized)
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
    }

    void 
    Window::imguiEnd()
    {
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }

    void
    Window::spinOnce()
    {
        glfwSwapBuffers(_internal_window);
        glfwPollEvents();
    }

} //namespace cupbr
