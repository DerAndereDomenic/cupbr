#include <Renderer/RayTracer.cuh>
#include <Core/KernelHelper.cuh>
#include <Core/Tracing.cuh>
#include <Geometry/Sphere.cuh>
#include <Geometry/Plane.cuh>

namespace cupbr
{
    namespace detail
    {
        __global__ void raytracing_kernel(Image<Vector3float> img,
            Scene scene,
            const Camera camera)
        {
            const uint32_t tid = ThreadHelper::globalThreadIndex();

            if (tid >= img.size())
            {
                return;
            }

            Ray ray = Tracing::launchRay(tid, img.width(), img.height(), camera);

            LocalGeometry geom = Tracing::traceRay(scene, ray);

            Vector3float inc_dir = Math::normalize(camera.position() - geom.P);

            //Lighting

            Vector3float radiance = 0;
            for (uint32_t i = 0; i < scene.light_count; ++i)
            {
                Vector3float lightDir = Math::normalize(scene.lights[i]->position - geom.P);
                Vector3float brdf = geom.material.brdf(geom.P, inc_dir, lightDir, geom.N);
                float d = Math::norm(geom.P - scene.lights[i]->position);
                Vector3float lightRadiance = scene.lights[i]->intensity / (d * d);
                float cosTerm = max(0.0f, Math::dot(geom.N, lightDir));


                //Shadow
                if (geom.depth != INFINITY)
                {
                    Ray shadow_ray(geom.P - EPSILON * ray.direction(), lightDir);

                    if (Tracing::traceVisibility(scene, d, shadow_ray))
                    {
                        radiance += brdf * lightRadiance * cosTerm;
                    }
                }
            }

            img[tid] = radiance;
        }
    } //namespace cupbr

    void
        PBRendering::raytracing(Scene& scene,
            const Camera& camera,
            Image<Vector3float>* output_img)
    {
        KernelSizeHelper::KernelSize config = KernelSizeHelper::configure(output_img->size());
        detail::raytracing_kernel << <config.blocks, config.threads >> > (*output_img, scene, camera);
        cudaSafeCall(cudaDeviceSynchronize());
    }

} //namespace cupbr