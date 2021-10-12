cmake_minimum_required(VERSION "3.8")

project(CUPBR LANGUAGES CXX C CUDA VERSION 1.0)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 14)

set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)

add_subdirectory(external/glfw)

set(BUILD_TESTING OFF)
add_subdirectory(external/tinyxml2)

file (GLOB_RECURSE cupbr_files LIST_DIRECTORIES false "lib/**.c*"
                                                      "lib/**.h*")

add_library(CUPBR STATIC "${cupbr_files}")

add_library(glad STATIC "external/glad/src/glad.c")

add_library(imgui STATIC "external/imgui/imgui.cpp"
                         "external/imgui/imgui_demo.cpp"
                         "external/imgui/imgui_draw.cpp"
                         "external/imgui/imgui_tables.cpp"
                         "external/imgui/imgui_widgets.cpp"
                         "external/imgui/backends/imgui_impl_glfw.cpp"
                         "external/imgui/backends/imgui_impl_opengl3.cpp")


add_executable(Renderer "src/source.cu")

if(MSVC)
	target_link_libraries(Renderer glfw OpenGL::GL tinyxml2 glad imgui CUPBR)
else()
    target_link_libraries(CUPBR -lpthread -lm OpenGL::GL tinyxml2 glad imgui)
	target_link_libraries(Renderer glfw CUPBR)
endif()
include_directories(lib/include 
                    external/glad/include 
                    external/glfw/include 
                    external/stb
                    external/imgui
                    external/tinyxml2
                    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_compile_definitions(CUPBR PUBLIC _USE_MATH_DEFINES)
target_compile_definitions(Renderer PUBLIC _USE_MATH_DEFINES)

set_target_properties(Renderer PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
