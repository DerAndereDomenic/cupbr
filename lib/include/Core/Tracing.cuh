#ifndef __CUPBR_CORE_TRACING_CUH
#define __CUPBR_CORE_TRACING_CUH

#include <Core/CUDA.cuh>
#include <Geometry/Ray.cuh>
#include <DataStructure/Camera.cuh>

/**
*   @brief A namespace that capsulate important ray tracing methods
*/
namespace Tracing
{
    /**
    *   @brief Maps a kernel thread index to an output ray
    *   @param[in] tid The kernel thread index
    *   @param[in] width The width of the output image
    *   @param[in] height The height of the output image
    *   @param[in] camera The camera
    *   @return A ray from the eye position through the corresponding pixel
    */
    __device__
    Ray
    launchRay(const uint32_t& tid, const uint32_t& width, const uint32_t& height, const Camera& camera);
}

#endif