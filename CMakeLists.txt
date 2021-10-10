cmake_minimum_required(VERSION "3.8")

project(CUPBR LANGUAGES CXX C CUDA VERSION 1.0)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)

add_subdirectory(external/glfw)

set(BUILD_TESTING OFF)
add_subdirectory(external/tinyxml2)

file (GLOB_RECURSE source_files LIST_DIRECTORIES false "src/**.c*"
                                                       "lib/**.c*"
                                                       "lib/**.h*")

set(source_files "${source_files}" "external/glad/src/glad.c"
                                   "external/imgui/imgui.cpp"
                                   "external/imgui/imgui_demo.cpp"
                                   "external/imgui/imgui_draw.cpp"
                                   "external/imgui/imgui_tables.cpp"
                                   "external/imgui/imgui_widgets.cpp"
                                   "external/imgui/backends/imgui_impl_glfw.cpp"
                                   "external/imgui/backends/imgui_impl_opengl3.cpp")


add_executable(${PROJECT_NAME} ${source_files})

target_link_libraries(${PROJECT_NAME} glfw -lpthread -lm ${GLEW_LIBRARIES} OpenGL::GL tinyxml2)
include_directories(lib/include 
                    external/glad/include 
                    external/glfw/include 
                    external/stb
                    external/imgui)

target_compile_definitions(${PROJECT_NAME} PUBLIC _USE_MATH_DEFINES)

set_target_properties(${PROJECT_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")