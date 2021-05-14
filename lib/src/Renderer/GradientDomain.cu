#include <Renderer/GradientDomain.cuh>
#include <Core/KernelHelper.cuh>
#include <Core/Tracing.cuh>
#include <Geometry/Sphere.cuh>
#include <Geometry/Plane.cuh>

namespace detail
{
    __device__ void
    traceImage( const uint32_t& tid,
                Scene& scene,
                const Camera& camera,
                const uint32_t& frameIndex,
                const uint32_t& maxTraceDepth,
                Image<Vector3float> img)
    {
        uint32_t seed = Math::tea<4>(tid, frameIndex);

        Ray ray = Tracing::launchRay(tid, img.width(), img.height(), camera, true, &seed);

        uint32_t trace_depth = 0;
        Vector3float radiance = 0;
        Vector3float rayweight = 1;
        Vector3float inc_dir, lightDir, lightRadiance;
        float d;

        Light light;

        do
        {
            //Direct illumination
            LocalGeometry geom = Tracing::traceRay(scene, ray);
            if(geom.depth == INFINITY)
            {
                if(scene.useEnvironmentMap)
                {
                    Vector2uint32_t pixel = Tracing::direction2UV(ray.direction(), scene.environment.width(), scene.environment.height());
                    radiance += rayweight * scene.environment(pixel);
                }
                break;
            }

            Vector3float normal = geom.N;

            inc_dir = Math::normalize(ray.origin() - geom.P);

            uint32_t useEnvironmentMap = scene.useEnvironmentMap ? 1 : 0;
            uint32_t light_sample = static_cast<uint32_t>(Math::rnd(seed) * (scene.light_count + useEnvironmentMap));

            if(light_sample != scene.light_count)
            {
                light = *(scene.lights[light_sample]); 

                switch(light.type)
                {
                    case LightType::POINT:
                    {
                        lightDir = Math::normalize(light.position - geom.P);
                        d = Math::norm(light.position - geom.P);
                        lightRadiance = light.intensity / (d*d);
                    }
                    break;
                    case LightType::AREA:
                    {
                        float xi1 = Math::rnd(seed) * 2.0f - 1.0f;
                        float xi2 = Math::rnd(seed) * 2.0f - 1.0f;

                        Vector3float sample = light.position + xi1 * light.halfExtend1 + xi2 * light.halfExtend2;
                        Vector3float n = Math::normalize(Math::cross(light.halfExtend1, light.halfExtend2));
                        float area = 4.0f*Math::norm(light.halfExtend1) * Math::norm(light.halfExtend2);

                        lightDir = Math::normalize(sample - geom.P);
                        d = Math::norm(sample - geom.P);

                        float NdotL = Math::dot(lightDir, n);
                        if(NdotL < 0) NdotL *= -1.0f;

                        float solidAngle =  area * NdotL / (d*d);

                        lightRadiance = light.radiance * solidAngle;
                    }
                    break;
                }
            }
            else // Use environment map
            {
                Vector4float sample = geom.material.sampleDirection(seed, inc_dir, geom.N);
                lightDir = Vector3float(sample);
                d = INFINITY; //TODO: Better way to do this
                Vector2uint32_t pixel = Tracing::direction2UV(lightDir, scene.environment.width(), scene.environment.height());
                lightRadiance = scene.environment(pixel)/sample.w;
            }
                
            Ray shadow_ray = Ray(geom.P + 0.01f*lightDir, lightDir);

            if(Tracing::traceVisibility(scene, d, shadow_ray))
            {
                radiance += (scene.light_count+useEnvironmentMap)*fmaxf(0.0f, Math::dot(normal,lightDir))*geom.material.brdf(geom.P,inc_dir,lightDir,normal)*lightRadiance*rayweight;
            }

            //Indirect illumination
            Vector4float direction_p = geom.material.sampleDirection(seed, inc_dir, geom.N);
            Vector3float direction = Vector3float(direction_p);
            rayweight = rayweight * fabs(Math::dot(direction, normal))*geom.material.brdf(geom.P, inc_dir, direction, normal)/direction_p.w;
                 
            ray = Ray(geom.P+0.01f*direction, direction);
            ++trace_depth;
        }while(trace_depth < maxTraceDepth);

        if(frameIndex > 0)
        {
            const float a = 1.0f/(static_cast<float>(frameIndex) + 1.0f);
            radiance = (1.0f-a)*img[tid] + a*radiance;
        }

        img[tid] = radiance;
    }

    __global__ void
    gradientdomain_kernel(Scene scene,
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

        traceImage(tid,
                   scene,
                   camera,
                   frameIndex,
                   maxTraceDepth,
                   img);
    }
}

void
PBRendering::gradientdomain(Scene& scene,
                            const Camera& camera,
                            const uint32_t& frameIndex,
                            const uint32_t& maxTraceDepth,
                            Image<Vector3float>* output_img)
{
    const KernelSizeHelper::KernelSize config = KernelSizeHelper::configure(output_img->size());
    detail::gradientdomain_kernel<<<config.blocks, config.threads>>>(scene, 
                                                                 camera,
                                                                 frameIndex,
                                                                 maxTraceDepth, 
                                                                 *output_img);
    cudaSafeCall(cudaDeviceSynchronize());
}
