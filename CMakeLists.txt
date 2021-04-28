cmake_minimum_required(VERSION "3.10")

project(CUPBR LANGUAGES CXX CUDA VERSION 1.0)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

find_package(GLEW REQUIRED)
set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(tinyxml2 REQUIRED)

file (GLOB_RECURSE source_files LIST_DIRECTORIES false "src/*.c*"
                                                       "lib/include/*/*.c*"
                                                       "lib/src/*/*.c*")

message("${source_files}")

add_executable(${PROJECT_NAME} ${source_files})

target_link_libraries(${PROJECT_NAME} glfw -lpthread -lm ${GLEW_LIBRARIES} OpenGL::GL tinyxml2)
include_directories(lib/include ${GLEW_INCLUDE_DIRS} ${GLFW_INCLUDE_DIRS})