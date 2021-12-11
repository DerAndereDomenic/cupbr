#include <Renderer/PathTracer.cuh>
#include <Core/KernelHelper.cuh>
#include <Core/Tracing.cuh>
#include <Geometry/Sphere.cuh>
#include <Geometry/Plane.cuh>

namespace cupbr
{
    namespace detail
    {
        struct RadiancePayload
        {
            uint32_t seed;
            Vector3float radiance = 0;
            Vector3float rayweight = 1;
            Vector3float out_dir;
            bool next_ray_valid = false;
        };

        inline __device__
        void emissiveIllumintation(Ray& ray, LocalGeometry& geom)
        {
            RadiancePayload* payload = ray.payload<RadiancePayload>();

            payload->radiance += payload->rayweight * geom.material.albedo_e;
        }

        inline __device__
        void directIllumination(Scene& scene, Ray& ray, LocalGeometry& geom, Vector3float& inc_dir)
        {
            //Direct illumination
            RadiancePayload* payload = ray.payload<RadiancePayload>();

            Vector3float normal = geom.N;

            //Don't shade back facing geometry
            if (geom.material.type != MaterialType::GLASS && Math::dot(normal, inc_dir) <= 0.0f)
            {
                payload->rayweight = 0;
                return;
            }

            uint32_t useEnvironmentMap = scene.useEnvironmentMap ? 1 : 0;
            uint32_t light_sample = static_cast<uint32_t>(Math::rnd(payload->seed) * (scene.light_count + useEnvironmentMap));

            Light light;
            Vector3float lightDir, lightRadiance;
            float d;
            if (light_sample != scene.light_count)
            {
                light = *(scene.lights[light_sample]);

                lightRadiance = light.sample(payload->seed, geom.P, lightDir, d);
            }
            else // Use environment map
            {
                Vector4float sample = geom.material.sampleDirection(payload->seed, inc_dir, geom.N);
                lightDir = Vector3float(sample);
                d = INFINITY; //TODO: Better way to do this
                Vector2uint32_t pixel = Tracing::direction2UV(lightDir, scene.environment.width(), scene.environment.height());
                lightRadiance = scene.environment(pixel) / sample.w;
            }

            Ray shadow_ray = Ray(geom.P + 0.001f * lightDir, lightDir);

            if (Tracing::traceVisibility(scene, d, shadow_ray))
            {
                payload->radiance += (scene.light_count + useEnvironmentMap) *
                    fmaxf(0.0f, Math::dot(normal, lightDir)) *
                    geom.material.brdf(geom.P, inc_dir, lightDir, normal) *
                    lightRadiance *
                    payload->rayweight;
            }
        }

        inline __device__
        void indirectIllumination(Ray& ray, LocalGeometry& geom, Vector3float& inc_dir)
        {
            //Indirect illumination
            RadiancePayload* payload = ray.payload<RadiancePayload>();


            Vector4float direction_p = geom.material.sampleDirection(payload->seed, inc_dir, geom.N);
            Vector3float direction = Vector3float(direction_p);
            if (Math::norm(direction) < EPSILON)
            {
                return;
            }
                
            payload->rayweight = payload->rayweight *
                fabs(Math::dot(direction, geom.N)) *
                geom.material.brdf(geom.P, inc_dir, direction, geom.N) / direction_p.w;
            payload->out_dir = direction;
            payload->next_ray_valid = true;
        }

        __global__ void
        pathtracer_kernel_nee(Scene scene,
                              const Camera camera,
                              const uint32_t frameIndex,
                              const uint32_t maxTraceDepth,
                              const bool useRussianRoulette,
                              Image<Vector3float> img)
        {
            const uint32_t tid = ThreadHelper::globalThreadIndex();

            if (tid >= img.size())
            {
                return;
            }

            uint32_t seed = Math::tea<4>(tid, frameIndex);

            Ray ray = Tracing::launchRay(tid, img.width(), img.height(), camera, true, &seed);
            RadiancePayload payload;
            payload.seed = seed;
            ray.setPayload(&payload);

            uint32_t trace_depth = 0;

            Light light;

            do
            {
                payload.next_ray_valid = false;
                LocalGeometry geom = Tracing::traceRay(scene, ray);
                if (geom.depth == INFINITY)
                {
                    if (scene.useEnvironmentMap)
                    {
                        Vector2uint32_t pixel = Tracing::direction2UV(ray.direction(), scene.environment.width(), scene.environment.height());
                        payload.radiance += payload.rayweight * scene.environment(pixel);
                    }
                    break;
                }
                Vector3float inc_dir = -1.0f * ray.direction();

                emissiveIllumintation(ray, geom);
                directIllumination(scene, ray, geom, inc_dir);

                // Russian Roulette
                if(useRussianRoulette)
                {
                    float alpha = Math::clamp(fmaxf(payload.rayweight.x, fmaxf(payload.rayweight.y, payload.rayweight.z)), 0.0f, 1.0f);
                    if(Math::rnd(payload.seed) > alpha || Math::safeFloatEqual(alpha, 0))
                    {
                        payload.next_ray_valid = false;
                        payload.rayweight = 0;
                        break;
                    }
                    payload.rayweight = payload.rayweight / alpha;
                }

                indirectIllumination(ray, geom, inc_dir);

                ray.traceNew(geom.P + 0.01f * payload.out_dir, payload.out_dir);

                if (!payload.next_ray_valid)break;
                ++trace_depth;
            } while (trace_depth < maxTraceDepth);

            if (frameIndex > 0)
            {
                const float a = 1.0f / (static_cast<float>(frameIndex) + 1.0f);
                ray.payload<RadiancePayload>()->radiance = (1.0f - a) * img[tid] + a * ray.payload<RadiancePayload>()->radiance;
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

            if (tid >= img.size())
            {
                return;
            }

            uint32_t seed = Math::tea<4>(tid, frameIndex);

            Ray ray = Tracing::launchRay(tid, img.width(), img.height(), camera, true, &seed);

            uint32_t traceDepth = static_cast<uint32_t>(Math::rnd(seed) * maxTraceDepth) + 1;
            uint32_t currentDepth = 0;
            float p = 1.0f / maxTraceDepth;
            bool firstMiss = false;

            Vector3float radiance = 1;
            Vector3float inc_dir;
            LocalGeometry geom;

            do
            {
                geom = Tracing::traceRay(scene, ray);
                inc_dir = Math::normalize(ray.origin() - geom.P);
                ++currentDepth;
                if (geom.depth == INFINITY || currentDepth >= traceDepth)
                {
                    if (geom.depth == INFINITY && currentDepth == 1)
                        firstMiss = true;
                    break;
                }

                Vector4float direction_p = geom.material.sampleDirection(seed, inc_dir, geom.N);
                Vector3float out_dir = Vector3float(direction_p);
                if (geom.material.type != MaterialType::GLASS)
                {
                    p *= direction_p.w;

                    radiance = radiance * geom.material.brdf(geom.P, inc_dir, out_dir, geom.N) * fmaxf(EPSILON, Math::dot(geom.N, out_dir));
                }

                ray = Ray(geom.P + 0.01f * out_dir, out_dir);

            } while (true);

            //Connect to light source
            if (!firstMiss)
            {
                Vector3float lightDir, lightRadiance;
                float d;

                uint32_t light_sample = static_cast<uint32_t>(Math::rnd(seed) * scene.light_count);
                Light light = *(scene.lights[light_sample]);

                lightRadiance = light.sample(seed, geom.P, lightDir, d);

                Ray shadow_ray = Ray(geom.P + 0.01f * lightDir, lightDir);

                if (Tracing::traceVisibility(scene, d, shadow_ray) && geom.depth != INFINITY)
                {
                    radiance = radiance * fmaxf(0.0f, Math::dot(geom.N, lightDir)) * geom.material.brdf(geom.P, inc_dir, lightDir, geom.N) * lightRadiance * static_cast<float>(scene.light_count);
                }
                else
                {
                    radiance = 0;
                }

                radiance = radiance / p;
            }
            else radiance = 0;


            if (frameIndex > 0)
            {
                const float a = 1.0f / (static_cast<float>(frameIndex) + 1.0f);
                radiance = (1.0f - a) * img[tid] + a * radiance;
            }

            img[tid] = radiance;
        }
    } //namespace detail

    void
    PBRendering::pathtracing(Scene& scene,
                             const Camera& camera,
                             const uint32_t& frameIndex,
                             const uint32_t& maxTraceDepth,
                             const bool& useRussianRoulette,
                             Image<Vector3float>* output_img)
    {
        const KernelSizeHelper::KernelSize config = KernelSizeHelper::configure(output_img->size());
        detail::pathtracer_kernel_nee << <config.blocks, config.threads >> > (scene,
                                                                              camera,
                                                                              frameIndex,
                                                                              maxTraceDepth,
                                                                              useRussianRoulette,
                                                                              *output_img);
        cudaSafeCall(cudaDeviceSynchronize());
    }

} //namespace cupbr
