#include <Renderer/RayTracer.cuh>
#include <Core/KernelHelper.cuh>
#include <Core/Tracing.cuh>
#include <Geometry/Sphere.cuh>
#include <Geometry/Plane.cuh>

namespace detail
{
    __global__ void raytracing_kernel(Image<Vector3float> img, 
                                      const Scene scene, 
                                      const uint32_t scene_size, 
                                      const Camera camera)
    {
        const uint32_t tid = ThreadHelper::globalThreadIndex();

        if(tid >= img.size())
        {
            return;
        }

        Ray ray = Tracing::launchRay(tid, img.width(), img.height(), camera);

        //Scene
        const Vector3float lightPos(0.0f,0.9f,2.0f);

        LocalGeometry geom = Tracing::traceRay(scene, scene_size, ray);

        Vector3float inc_dir = Math::normalize(camera.position() - geom.P);
        Vector3float lightDir = Math::normalize(lightPos - geom.P);


        //Lighting

        Vector3float brdf = geom.material.brdf(geom.P, inc_dir, lightDir, geom.N);
        Vector3float lightIntensity = Vector3float(10,10,10); //White light
        float d = Math::norm(geom.P-lightPos);
        Vector3float lightRadiance = lightIntensity/(d*d);
        float cosTerm = max(0.0f,Math::dot(geom.N, lightDir));
        Vector3float radiance = brdf*lightRadiance*cosTerm;

        //Shadow
        if(geom.depth != INFINITY)
        {
            Ray shadow_ray(geom.P-EPSILON*ray.direction(), lightDir);
            
            if(!Tracing::traceVisibility(scene, scene_size, d, shadow_ray))
            {
                radiance = 0;
            }
        }

        img[tid] = radiance;
    }
}

void
PBRendering::raytracing(const Scene scene,
                        const uint32_t& scene_size,
                        const Camera& camera,
                        Image<Vector3float>* output_img)
{
    KernelSizeHelper::KernelSize config = KernelSizeHelper::configure(output_img->size());
    detail::raytracing_kernel<<<config.blocks, config.threads>>>(*output_img,scene,scene_size,camera);
    cudaSafeCall(cudaDeviceSynchronize());
}