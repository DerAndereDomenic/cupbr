#include <Renderer/PathTracer.cuh>
#include <Core/KernelHelper.cuh>
#include <Core/Tracing.cuh>
#include <Geometry/Sphere.cuh>
#include <Geometry/Plane.cuh>

namespace detail
{
    __global__ void
    pathtracer_kernel(const Scene scene,
                      const uint32_t scene_size,
                      const Camera camera,
                      const uint32_t frameIndex,
                      const uint32_t maxTraceDepth,
                      Image<Vector3float> img)
    {
        const uint32_t tid = ThreadHelper::globalThreadIndex();

        if(tid >= img.size())
        {
            return;
        }

        uint32_t seed = Math::tea<4>(tid, frameIndex);

        Ray ray = Tracing::launchRay(tid, img.width(), img.height(), camera);
        
        //Scene
        Vector3float lightPos(0.0f,0.9f,2.0f);

        uint32_t trace_depth = 0;
        Vector3float radiance = 0;
        Vector3float rayweight = 1;
        Vector3float direction = 0;
        bool continueTracing;
        float p;

        do
        {
            continueTracing = false;

            //Direct illumination
            LocalGeometry geom = Tracing::traceRay(scene, scene_size, ray);
            if(geom.depth == INFINITY)break;
            Vector3float normal = geom.N;

            Vector3float inc_dir = Math::normalize(ray.origin() - geom.P);
            Vector3float lightDir = Math::normalize(lightPos - geom.P);
            float d = Math::norm(lightPos - geom.P);
            Vector3float lightRadiance = Vector3float(1.0f) / (d*d);

            Ray shadow_ray = Ray(geom.P + 0.01f*lightDir, lightDir);

            if(Tracing::traceVisibility(scene, scene_size, d, shadow_ray))
            {
                radiance += fmaxf(0.0f, Math::dot(normal,lightDir))*geom.material.brdf(geom.P,inc_dir,lightDir,normal)*lightRadiance*rayweight;
            }

            //Indirect illumination
            switch(geom.material.type)
            {
                case PHONG:
                case LAMBERT:
                {
                    const float xi_1 = Math::rnd(seed);
                    const float xi_2 = Math::rnd(seed);

                    const float r = sqrtf(xi_1);
                    const float phi = 2.0f*3.14159f*xi_2;

                    const float x = r*cos(phi);
                    const float y = r*sin(phi);
                    const float z = sqrtf(fmaxf(0.0f, 1.0f - x*x-y*y));

                    direction = Math::normalize(Math::toLocalFrame(normal, Vector3float(x,y,z)));

                    p = fmaxf(EPSILON, Math::dot(direction, normal))/3.14159f;

                    rayweight = rayweight * fmaxf(EPSILON, Math::dot(direction, normal))*geom.material.brdf(geom.P, inc_dir, direction, normal)/p;
                    continueTracing = true;
                }
                break;
                /*case PHONG:
                {

                }
                break;*/
                case MIRROR:
                {
                    direction = Math::reflect(inc_dir, normal);

                    p = 1.0f;

                    rayweight = rayweight * fmaxf(EPSILON, Math::dot(direction, normal))*geom.material.brdf(geom.P, inc_dir, direction, normal)/p;
                    continueTracing = true;
                }
                break;
                case GLASS:
                {
                    const float NdotV = Math::dot(inc_dir, geom.N);
                    bool outside = NdotV > 0.0f;
                    float eta = outside ? 1.0f/geom.material.eta : geom.material.eta;
                    Vector3float normal = outside ? geom.N : -1.0f*geom.N;
                    float F0 = outside ? (1.0f - geom.material.eta) / (1.0f + geom.material.eta) : (-1.0f + geom.material.eta) / (1.0f + geom.material.eta);
                    F0 *= F0;

                    float p_reflect = Math::fresnel_schlick(F0, Math::dot(inc_dir, normal));
                    float xi = Math::rnd(seed);

                    Vector3float refraction_dir = Math::refract(eta, inc_dir, normal);
                    if(xi <= p_reflect || Math::safeFloatEqual(Math::norm(refraction_dir), 0.0f))
                    {
                        direction = Math::reflect(inc_dir, normal);
                    }
                    else
                    {
                        rayweight = Math::dot(normal, refraction_dir)/Math::dot(normal, inc_dir) * rayweight;
                        direction = refraction_dir;
                    }
                    
                    continueTracing = true;
                }
                break;
            }
            ray = Ray(geom.P+0.01f*direction, direction);

            ++trace_depth;
        }while(trace_depth < maxTraceDepth && continueTracing);

        if(frameIndex > 0)
        {
            const float a = 1.0f/(static_cast<float>(frameIndex) + 1.0f);
            radiance = (1.0f-a)*img[tid] + a*radiance;
        }

        img[tid] = radiance;
    }
}

void
PBRendering::pathtracing(const Scene scene,
                         const uint32_t& scene_size,
                         const Camera& camera,
                         const uint32_t& frameIndex,
                         const uint32_t& maxTraceDepth,
                         Image<Vector3float>* output_img)
{
    const KernelSizeHelper::KernelSize config = KernelSizeHelper::configure(output_img->size());
    detail::pathtracer_kernel<<<config.blocks, config.threads>>>(scene, 
                                                                 scene_size, 
                                                                 camera,
                                                                 frameIndex,
                                                                 maxTraceDepth, 
                                                                 *output_img);
    cudaSafeCall(cudaDeviceSynchronize());
}
