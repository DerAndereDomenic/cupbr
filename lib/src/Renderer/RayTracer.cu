#include <Renderer/RayTracer.cuh>
#include <Core/KernelHelper.cuh>
#include <Core/Tracing.cuh>
#include <Geometry/Sphere.cuh>
#include <Geometry/Plane.cuh>

namespace detail
{
    __global__ void raytracing_kernel(Image<Vector3float> img, 
                                      const Scene scene, 
                                      const Camera camera)
    {
        const uint32_t tid = ThreadHelper::globalThreadIndex();

        if(tid >= img.size())
        {
            return;
        }

        Ray ray = Tracing::launchRay(tid, img.width(), img.height(), camera);

        LocalGeometry geom = Tracing::traceRay(scene, ray);

        Vector3float inc_dir = Math::normalize(camera.position() - geom.P);
        Vector3float lightDir = Math::normalize(scene.lights[0]->position - geom.P);


        //Lighting

        Vector3float brdf = geom.material.brdf(geom.P, inc_dir, lightDir, geom.N);
        float d = Math::norm(geom.P-scene.lights[0]->position);
        Vector3float lightRadiance = scene.lights[0]->intensity/(d*d);
        float cosTerm = max(0.0f,Math::dot(geom.N, lightDir));
        Vector3float radiance = brdf*lightRadiance*cosTerm;

        //Shadow
        if(geom.depth != INFINITY)
        {
            Ray shadow_ray(geom.P-EPSILON*ray.direction(), lightDir);
            
            if(!Tracing::traceVisibility(scene, d, shadow_ray))
            {
                radiance = 0;
            }
        }

        img[tid] = radiance;
    }
}

void
PBRendering::raytracing(const Scene scene,
                        const Camera& camera,
                        Image<Vector3float>* output_img)
{
    KernelSizeHelper::KernelSize config = KernelSizeHelper::configure(output_img->size());
    detail::raytracing_kernel<<<config.blocks, config.threads>>>(*output_img,scene,camera);
    cudaSafeCall(cudaDeviceSynchronize());
}