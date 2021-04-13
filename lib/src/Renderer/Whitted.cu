#include <Renderer/Whitted.cuh>
#include <Core/KernelHelper.cuh>
#include <Core/Tracing.cuh>
#include <Geometry/Sphere.cuh>
#include <Geometry/Plane.cuh>

namespace detail
{
    __global__ void
    whitted_kernel(const Scene scene,
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

        Ray ray = Tracing::launchRay(tid, img.width(), img.height(), camera);

        //Scene
        const Vector3float lightPos(0.0f, 0.9f, 2.0f);

        Vector3float total_radiance = 0;
        Vector3float radiance = 1.0f;
        float reflection = 0.01f;
        float lightFactor;

        for(uint32_t i = 0; i < maxTraceDepth; ++i)
        {
            //Direct illumination
            LocalGeometry geom = Tracing::traceRay(scene, scene_size, ray);

            Vector3float inc_dir = Math::normalize(ray.origin() - geom.P);
            Vector3float lightDir = Math::normalize(lightPos - geom.P);
            bool continueTracing = true;

            if(Math::dot(inc_dir,geom.N) < 0.0f)break;

            //Lighting
            switch(geom.material.type)
            {
                case MaterialType::LAMBERT:
                case MaterialType::PHONG:
                {
                    Vector3float brdf = geom.material.brdf(geom.P, inc_dir, lightDir, geom.N);
                    Vector3float lightIntensity = Vector3float(10,10,10); //White light
                    float d = Math::norm(geom.P-lightPos);
                    Vector3float lightRadiance = lightIntensity/(d*d);
                    float cosTerm = max(0.0f,Math::dot(geom.N, lightDir));

                    //Shadow
                    lightFactor = 1.0f;
                    if(geom.depth != INFINITY)
                    {
                        Ray shadow_ray(geom.P-EPSILON*ray.direction(), lightDir);
                            
                        if(!Tracing::traceVisibility(scene, scene_size, d, shadow_ray))
                        {
                            lightFactor = 0.0f;
                        }
                    }

                    radiance = radiance*lightFactor*brdf*lightRadiance*cosTerm;
                    total_radiance += radiance;

                    continueTracing = false;
                }
                if(geom.material.type == MaterialType::LAMBERT)
                {
                    break;
                }
                radiance = reflection;
                reflection /= 10.0f;
                case MaterialType::MIRROR:
                {
                    Vector3float reflected = Math::reflect(inc_dir, geom.N);
                    ray = Ray(geom.P+EPSILON*reflected, reflected);

                    radiance = radiance*geom.material.brdf(geom.P, inc_dir, reflected, geom.N);
                    continueTracing = true;
                }
                break;
            }
            if(!continueTracing)break;
        }
        
    
        img[tid] = total_radiance;
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