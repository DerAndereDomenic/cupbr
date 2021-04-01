cmake_minimum_required(VERSION "3.10")

project(CUPBR LANGUAGES CXX CUDA VERSION 1.0)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

find_package(GLEW REQUIRED)
set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)

add_executable(${PROJECT_NAME} src/source.cu)

target_link_libraries(${PROJECT_NAME} glfw -lpthread -lm ${GLEW_LIBRARIES} OpenGL::GL)
include_directories(lib/include ${GLEW_INCLUDE_DIRS} ${GLFW_INCLUDE_DIRS})
message("${GLFW_INCLUDE_DIRS}")