#include <Renderer/PathTracer.cuh>
#include <Core/KernelHelper.cuh>
#include <Core/Tracing.cuh>
#include <Geometry/Sphere.cuh>
#include <Geometry/Plane.cuh>

namespace detail
{
    struct RadiancePayload
    {
        Vector3float radiance = 0;
        Vector3float rayweight = 1;
    };

    __global__ void
    pathtracer_kernel_nee(Scene scene,
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

        Ray ray = Tracing::launchRay(tid, img.width(), img.height(), camera, true, &seed);
        RadiancePayload payload;
        ray.setPayload(&payload);

        uint32_t trace_depth = 0;
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
                    ray.payload<RadiancePayload>()->radiance += ray.payload<RadiancePayload>()->rayweight * scene.environment(pixel);
                }
                break;
            }

            Vector3float normal = geom.N;

            inc_dir = Math::normalize(ray.origin() - geom.P);

            //Don't shade back facing geometry
            if(geom.material.type != GLASS && Math::dot(normal, inc_dir) <= 0.0f)
            {
                break;
            }

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
                ray.payload<RadiancePayload>()->radiance += (scene.light_count+useEnvironmentMap) *
                                                            fmaxf(0.0f, Math::dot(normal,lightDir)) *
                                                            geom.material.brdf(geom.P,inc_dir,lightDir,normal) *
                                                            lightRadiance *
                                                            ray.payload<RadiancePayload>()->rayweight;
            }

            //Indirect illumination
            Vector4float direction_p = geom.material.sampleDirection(seed, inc_dir, geom.N);
            Vector3float direction = Vector3float(direction_p);
            ray.payload<RadiancePayload>()->rayweight = ray.payload<RadiancePayload>()->rayweight * 
                                                        fabs(Math::dot(direction, normal)) * 
                                                        geom.material.brdf(geom.P, inc_dir, direction, normal)/direction_p.w;
                 
            ray.traceNew(geom.P+0.01f*direction, direction);
            ++trace_depth;
        }while(trace_depth < maxTraceDepth);

        if(frameIndex > 0)
        {
            const float a = 1.0f/(static_cast<float>(frameIndex) + 1.0f);
            ray.payload<RadiancePayload>()->radiance = (1.0f-a)*img[tid] + a*ray.payload<RadiancePayload>()->radiance;
        }

        img[tid] = ray.payload<RadiancePayload>()->radiance;
    }

    __global__ void
    pathtracer_kernel(Scene scene,
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

        Ray ray = Tracing::launchRay(tid, img.width(), img.height(), camera, true, &seed);

        uint32_t traceDepth = static_cast<uint32_t>(Math::rnd(seed) * maxTraceDepth) + 1;
        uint32_t currentDepth = 0;
        float p = 1.0f/maxTraceDepth;
        bool firstMiss = false;

        Vector3float radiance = 1;
        Vector3float inc_dir;
        LocalGeometry geom;

        do
        {
            geom = Tracing::traceRay(scene, ray);
            inc_dir = Math::normalize(ray.origin() - geom.P);
            ++currentDepth;
            if(geom.depth == INFINITY || currentDepth >= traceDepth)
            {
                if(geom.depth == INFINITY && currentDepth == 1)
                    firstMiss = true;
                break;
            }

            Vector4float direction_p = geom.material.sampleDirection(seed, inc_dir, geom.N);
            Vector3float out_dir = Vector3float(direction_p);
            if(geom.material.type != GLASS)
            {
                p *= direction_p.w;

                radiance = radiance*geom.material.brdf(geom.P, inc_dir, out_dir, geom.N) * fmaxf(EPSILON, Math::dot(geom.N, out_dir));
            }

            ray = Ray(geom.P + 0.01f*out_dir, out_dir);

        }while(true);

        //Connect to light source
        if(!firstMiss)
        {
            Vector3float lightDir, lightRadiance;
            float d;

            uint32_t light_sample = static_cast<uint32_t>(Math::rnd(seed) * scene.light_count);
            Light light = *(scene.lights[light_sample]); 

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

            Ray shadow_ray = Ray(geom.P + 0.01f*lightDir, lightDir);

            if(Tracing::traceVisibility(scene, d, shadow_ray) && geom.depth != INFINITY)
            {
                radiance = radiance*fmaxf(0.0f, Math::dot(geom.N,lightDir))*geom.material.brdf(geom.P,inc_dir,lightDir,geom.N)*lightRadiance*static_cast<float>(scene.light_count);
            }
            else
            {
                radiance = 0;
            }

            radiance = radiance/p;
        }
        else radiance = 0;
        

        if(frameIndex > 0)
        {
            const float a = 1.0f/(static_cast<float>(frameIndex) + 1.0f);
            radiance = (1.0f-a)*img[tid] + a*radiance;
        }

        img[tid] = radiance;
    }
}

void
PBRendering::pathtracing(Scene& scene,
                         const Camera& camera,
                         const uint32_t& frameIndex,
                         const uint32_t& maxTraceDepth,
                         Image<Vector3float>* output_img)
{
    const KernelSizeHelper::KernelSize config = KernelSizeHelper::configure(output_img->size());
    detail::pathtracer_kernel_nee<<<config.blocks, config.threads>>>(scene, 
                                                                 camera,
                                                                 frameIndex,
                                                                 maxTraceDepth, 
                                                                 *output_img);
    cudaSafeCall(cudaDeviceSynchronize());
}
