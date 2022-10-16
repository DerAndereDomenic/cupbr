#include <Interaction/MousePicker.h>
#include <Core/Tracing.h>

namespace cupbr
{
    namespace detail
    {
        CUPBR_GLOBAL void
        pickMouse_kernel(const uint32_t x,
                         const uint32_t y,
                         const uint32_t width,
                         const uint32_t height,
                         GeometryScene scene,
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
                           Scene* scene,
                           Camera& camera,
                           int32_t* outSceneIndex)
    {
        GeometryScene* geom_scene = dynamic_cast<GeometryScene*>(scene);
        if (geom_scene == nullptr)
        {
            std::cerr << "ERROR: MousePicker received scene that does not hold geometry information!\n";
            return;
        }

        detail::pickMouse_kernel << <1, 1 >> > (x,
                                                y,
                                                width,
                                                height,
                                                *geom_scene,
                                                camera,
                                                outSceneIndex);
        synchronizeDefaultStream();
    }

} //namespace cupbr