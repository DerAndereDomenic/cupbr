#include <GL/Window.h>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

#include <glad/glad.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

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
