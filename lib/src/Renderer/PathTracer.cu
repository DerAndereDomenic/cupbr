#include <Renderer/PathTracer.cuh>
#include <Core/KernelHelper.cuh>
#include <Core/Tracing.cuh>
#include <Geometry/Sphere.cuh>
#include <Geometry/Plane.cuh>

namespace detail
{
    __global__ void
    pathtracer_kernel(const Scene scene,
                      const uint32_t scene_size,
                      const Camera camera,
                      const uint32_t maxTraceDepth,
                      Image<Vector3float> img)
    {
        const uint32_t tid = ThreadHelper::globalThreadIndex();

        if(tid >= img.size())
        {
            return;
        }
    }
}

void
PBRendering::pathtracing(const Scene scene,
                         const uint32_t& scene_size,
                         const Camera& camera,
                         const uint32_t& maxTraceDepth,
                         Image<Vector3float>* output_img)
{
    const KernelSizeHelper::KernelSize config = KernelSizeHelper::configure(output_img->size());
    detail::pathtracer_kernel<<<config.blocks, config.threads>>>(scene, 
                                                                 scene_size, 
                                                                 camera, 
                                                                 maxTraceDepth, 
                                                                 *output_img);
    cudaSafeCall(cudaDeviceSynchronize());
}
