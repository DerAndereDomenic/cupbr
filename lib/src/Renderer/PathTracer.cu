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
        Vector3float brdf = 0;
        Vector3float rayweight = 1;
        bool continueTracing;

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
            Vector3float lightRadiance = Vector3float(10.0f) / (d*d);

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

                    Vector3float direction = Math::normalize(Math::toLocalFrame(normal, Vector3float(x,y,z)));
                    ray = Ray(geom.P + 0.01f*direction, direction);

                    float p = fmaxf(EPSILON, Math::dot(direction, normal))/3.14159f;
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

                }
                break;
                case GLASS:
                {

                }
                break;
            }

            ++trace_depth;
        }while(trace_depth < maxTraceDepth && continueTracing);

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
