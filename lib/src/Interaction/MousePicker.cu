#include <Interaction/MousePicker.cuh>
#include <Core/Tracing.cuh>

namespace detail
{
    __global__ void
    pickMouse_kernel(const uint32_t x,
                     const uint32_t y,
                     const uint32_t width,
                     const uint32_t height,
                     Scene scene,
                     const Camera camera,
                     Material* material)
    {
        const Vector2uint32_t pixel(x,y);
        Ray ray = Tracing::launchRay(pixel, width, height, camera);

        LocalGeometry geom = Tracing::traceRay(scene, ray);

        if(geom.depth != INFINITY)
        {
            material->type = geom.material.type;
            material->albedo_d = geom.material.albedo_d;
            material->albedo_s = geom.material.albedo_s;
            material->shininess = geom.material.shininess;
            material->eta = geom.material.eta;
        }
    }
}

void 
Interaction::pickMouse(const uint32_t& x,
                       const uint32_t& y,
                       const uint32_t& width,
                       const uint32_t& height,
                       Scene& scene,
                       Camera& camera,
                       Material* outMaterial)
{
    detail::pickMouse_kernel<<<1,1>>>(x,
                                      y,
                                      width,
                                      height,
                                      scene,
                                      camera,
                                      outMaterial);
    cudaSafeCall(cudaDeviceSynchronize());
}