#include <Interaction/MousePicker.h>
#include <Core/Tracing.h>

namespace cupbr
{
    namespace detail
    {
        __global__ void
        pickMouse_kernel(const uint32_t x,
                         const uint32_t y,
                         const uint32_t width,
                         const uint32_t height,
                         Scene scene,
                         const Camera camera,
                         int32_t* scene_index)
        {
            const Vector2uint32_t pixel(x, y);
            Ray ray = Tracing::launchRay(pixel, width, height, camera);

            LocalGeometry geom = Tracing::traceRay(scene, ray);
            if(geom.depth != INFINITY)
                *(scene_index) = geom.scene_index;
        }
    } //namespace detail

    void
    Interaction::pickMouse(const uint32_t& x,
                           const uint32_t& y,
                           const uint32_t& width,
                           const uint32_t& height,
                           Scene& scene,
                           Camera& camera,
                           int32_t* outSceneIndex)
    {
        detail::pickMouse_kernel << <1, 1 >> > (x,
                                                y,
                                                width,
                                                height,
                                                scene,
                                                camera,
                                                outSceneIndex);
        synchronizeDefaultStream();
    }

} //namespace cupbr