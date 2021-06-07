#include <Interaction/MousePicker.cuh>

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
        //Ortographic projection
        const float ratio_x = 2.0f*(static_cast<float>(x) / static_cast<float>(width) - 0.5f);
        const float ratio_y = 2.0f*(static_cast<float>(y) / static_cast<float>(height) - 0.5f);

        const Vector3float world_pos = camera.position() + camera.zAxis() + ratio_x * camera.xAxis() + ratio_y * camera.yAxis();

        Ray ray(world_pos, camera.zAxis());

        LocalGeometry geom = traceRay(scene, ray);

        if(geom.depth != INFINITY)
        {
            material = geom.material;
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

}