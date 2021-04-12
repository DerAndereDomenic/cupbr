#include <Renderer/Whitted.cuh>

namespace detail
{
    __global__ void
    whitted_kernel(const Scene scene,
                   const uint32_t scene_size,
                   const Camera camera,
                   const uint32_t maxTraceDepth,
                   Image<Vector3float> output_img)
    {

    }
}

void
PBRendering::whitted(const Scene scene,
                     const uint32_t& scene_size,
                     const Camera& camera,
                     const uint32_t& maxTraceDepth,
                     Image<Vector3float>* output_img)
{
    KernelSizeHelper::KernelSize config = KernelSizeHelper::configure(output_img->size());
    detail::whitted_kernel<<<config.blocks, config.threads>>>(scene,
                                                              scene_size,
                                                              camera,
                                                              maxTraceDepth,
                                                              *output_img);
    cudaSafeCall(cudaDeviceSynchronize());
}