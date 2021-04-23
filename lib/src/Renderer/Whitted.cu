#include <Renderer/Whitted.cuh>
#include <Core/KernelHelper.cuh>
#include <Core/Tracing.cuh>
#include <Geometry/Sphere.cuh>
#include <Geometry/Plane.cuh>

namespace detail
{
    __device__ Vector3float
    estimateRadiance(const Ray& ray, 
                     const Scene scene, 
                     const uint32_t& traceDepth, 
                     const uint32_t maxTraceDepth)
    {
        if(traceDepth >= maxTraceDepth)
        {
            return Vector3float(0);
        }

        //Scene
        const Vector3float lightPos(0.0f, 0.9f, 2.0f);

        Vector3float radiance = 0;
        //float reflection = 0.01f;
        float lightFactor;
        
        LocalGeometry geom = Tracing::traceRay(scene, ray);

        Vector3float inc_dir = Math::normalize(ray.origin() - geom.P);
        Vector3float lightDir = Math::normalize(lightPos - geom.P);

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

                    if(!Tracing::traceVisibility(scene, d, shadow_ray))
                    {
                        lightFactor = 0.0f;
                    }
                }
                radiance = lightFactor*brdf*lightRadiance*cosTerm;
            }
            break;
            /*if(geom.material.type == MaterialType::LAMBERT)
            {
                break;
            }
            radiance = reflection;
            reflection /= 10.0f;*/
            case MaterialType::MIRROR:
            {
                Vector3float reflected = Math::reflect(inc_dir, geom.N);
                Ray new_ray = Ray(geom.P+EPSILON*reflected, reflected);

                radiance = estimateRadiance(new_ray,
                                            scene,
                                            traceDepth+1,
                                            maxTraceDepth)*geom.material.brdf(geom.P, inc_dir, reflected, geom.N)*max(0.0f,Math::dot(geom.N, reflected));
            }
            break;
            case MaterialType::GLASS:
            {
                bool outside = Math::dot(inc_dir,geom.N) > 0;
                float eta = outside ? 1.0f/geom.material.eta : geom.material.eta;
                Vector3float normal = outside ? geom.N : -1.0f*geom.N;
                float F0 = outside ? (1.0f - geom.material.eta) / (1.0f + geom.material.eta) : (-1.0f + geom.material.eta) / (1.0f + geom.material.eta);
                F0 *= F0;

                float F = Math::fresnel_schlick(F0, Math::dot(inc_dir, normal));

                Vector3float refracted = Math::refract(eta, inc_dir, normal);
                Vector3float reflected = Math::reflect(inc_dir, normal);

                if(!Math::safeFloatEqual(Math::norm(refracted), 0.0f))
                {
                    radiance += (1.0f - F) * estimateRadiance(Ray(geom.P + 0.01f*refracted, refracted),
                                                                  scene,
                                                                  traceDepth+1,
                                                                  maxTraceDepth);//*geom.material.brdf(geom.P, inc_dir, refracted, geom.N)*max(0.0f, Math::dot(geom.N,refracted));
                }
                
                radiance += (F) * estimateRadiance(Ray(geom.P + 0.01f*reflected, reflected),
                                                          scene,
                                                          traceDepth+1,
                                                          maxTraceDepth);//*geom.material.brdf(geom.P, inc_dir, reflected, geom.N)*max(0.0f, Math::dot(geom.N,reflected));
                    
            }
            break;
        }

        return radiance;
    }

    __global__ void
    whitted_kernel(const Scene scene,
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

        img[tid] = estimateRadiance(ray,
                                    scene,
                                    0,
                                    maxTraceDepth);
    }
}

void
PBRendering::whitted(const Scene scene,
                     const Camera& camera,
                     const uint32_t& maxTraceDepth,
                     Image<Vector3float>* output_img)
{
    KernelSizeHelper::KernelSize config = KernelSizeHelper::configure(output_img->size());
    detail::whitted_kernel<<<config.blocks, config.threads>>>(scene,
                                                              camera,
                                                              maxTraceDepth,
                                                              *output_img);
    cudaSafeCall(cudaDeviceSynchronize());
}