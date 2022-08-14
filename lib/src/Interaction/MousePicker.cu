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
            *(scene_index) = geom.scene_index;
        }

        __global__ void
        updateMaterial_kernel(Scene scene,
                              int32_t* scene_index,
                              Material* newMaterial)
        {
            /*Geometry* element = scene[*scene_index];
            element->material->type = newMaterial->type;
            element->material->albedo_e = newMaterial->albedo_e;
            element->material->albedo_d = newMaterial->albedo_d;
            element->material->albedo_s = newMaterial->albedo_s;
            element->material->shininess = newMaterial->shininess;
            element->material->eta = newMaterial->eta;
            element->material->roughness = newMaterial->roughness;
            element->material->volume = newMaterial->volume;*/
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
        cudaSafeCall(cudaDeviceSynchronize());
    }

    void
    Interaction::updateMaterial(Scene& scene,
                                int32_t* scene_index,
                                Material* newMaterial)
    {
        detail::updateMaterial_kernel << <1, 1 >> > (scene, scene_index, newMaterial);
        cudaSafeCall(cudaDeviceSynchronize());
    }

} //namespace cupbr