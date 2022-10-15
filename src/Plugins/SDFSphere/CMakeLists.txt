set(target_name SDFSphere)

add_library(${target_name} SHARED ${target_name}.cu)

include_directories(${PROJECT_SOURCE_DIR}/lib/include 
                    ${PROJECT_SOURCE_DIR}/external/glad/include 
                    ${PROJECT_SOURCE_DIR}/external/glfw/include 
                    ${PROJECT_SOURCE_DIR}/external/stb
                    ${PROJECT_SOURCE_DIR}/external/imgui
                    ${PROJECT_SOURCE_DIR}/external/tinyxml2
                    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_compile_definitions(${target_name} PUBLIC _USE_MATH_DEFINES)
set_target_properties(${target_name} PROPERTIES CUDA_ARCHITECTURES "72"
						C_STANDARD 99
						CXX_STANDARD 17
						CUDA_STANDARD 17)

string(APPEND CMAKE_CUDA_FLAGS " -use_fast_math")