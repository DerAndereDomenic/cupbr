#include <CUPBR.h>

namespace cupbr
{

    namespace detail
    {
        struct RadiancePayload
        {
            uint32_t seed;
            Vector3float radiance = 0;
            Vector3float rayweight = 1;
            Vector3float ray_start;
            Vector3float out_dir;
            bool next_ray_valid;
        };

        CUPBR_DEVICE void directIllumination(Scene& scene, Ray& ray, LocalGeometry& geom, Vector3float& inc_dir)
        {
            //Direct illumination
            RadiancePayload* payload = ray.payload<RadiancePayload>();

            Vector3float normal = geom.N;

            //Don't shade back facing geometry
            if (geom.material->type != MaterialType::REFRACTIVE && Math::dot(normal, inc_dir) <= 0.0f)
            {
                //payload->rayweight = 0;
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
                Vector4float sample = geom.material->sampleDirection(payload->seed, inc_dir, geom.N);
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
                    geom.material->brdf(geom.P, inc_dir, lightDir, normal) *
                    lightRadiance *
                    payload->rayweight;
            }
        }

        CUPBR_DEVICE void indirectIllumination(Ray& ray, LocalGeometry& geom, Vector3float& inc_dir)
        {
            RadiancePayload* payload = ray.payload<RadiancePayload>();

            //Kind of back-face culling to handle back facing shading normals
            if (geom.material->type != MaterialType::REFRACTIVE && Math::dot(geom.N, inc_dir) <= 0.0f)
            {
                payload->ray_start = geom.P;
                payload->out_dir = -1.0f * inc_dir;
                payload->next_ray_valid = true;
                return;
            }

            //Indirect illumination
            Vector4float direction_p = geom.material->sampleDirection(payload->seed, inc_dir, geom.N);
            Vector3float direction = Vector3float(direction_p);
            if (Math::norm(direction) == 0)
                return;
            ray.payload<RadiancePayload>()->rayweight = ray.payload<RadiancePayload>()->rayweight *
                fabs(Math::dot(direction, geom.N)) *
                geom.material->brdf(geom.P, inc_dir, direction, geom.N) / direction_p.w;
            payload->out_dir = direction;
            payload->ray_start = geom.P;
            payload->next_ray_valid = true;
        }

        CUPBR_GLOBAL void
        volume_kernel(Scene scene,
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
                LocalGeometry geom;
                    
                geom = Tracing::traceRay(scene, ray);

                Vector3float inc_dir = -1.0f * ray.direction(); //Points away from surface
                if (geom.depth == INFINITY)
                {
                    if (scene.useEnvironmentMap)
                    {
                        Vector2uint32_t pixel = Tracing::direction2UV(ray.direction(), scene.environment.width(), scene.environment.height());
                        payload.radiance += payload.rayweight * scene.environment(pixel);
                    }
                    break;
                }

                directIllumination(scene, ray, geom, inc_dir);
                indirectIllumination(ray, geom, inc_dir);

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
                    
                ray.traceNew(payload.ray_start + 0.001f * payload.out_dir, payload.out_dir);
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

        CUPBR_GLOBAL void
        generate_kernel(const Camera camera,
                        const uint32_t frameIndex,
                        const uint32_t numPaths,
                        const Image<Vector3float> img,
                        Ray* ray_buffer,
                        RadiancePayload* payloads)
        {
            const uint32_t tid = ThreadHelper::globalThreadIndex();

            if(tid >= numPaths)
            {
                return;
            }

            uint32_t seed = Math::tea<4>(tid, frameIndex);

            ray_buffer[tid] = Tracing::launchRay(tid, img.width(), img.height(), camera, true, &seed);
            payloads[tid] = RadiancePayload();
            payloads[tid].seed = seed;
            ray_buffer[tid].setPayload(payloads + tid);
        }

        CUPBR_GLOBAL void
        extend_kernel(Scene scene,
                      const uint32_t numPaths,
                      Ray* ray_buffer,
                      LocalGeometry* geom,
                      int* activePaths)
        {
            uint32_t tid = ThreadHelper::globalThreadIndex();

            if(tid >= numPaths)
            {
                return;
            }

            *activePaths = 0;

            geom[tid] = Tracing::traceRay(scene, ray_buffer[tid]);
        }

        CUPBR_GLOBAL void
        shade_kernel(Scene scene,
                     const uint32_t numPaths,
                     Ray* ray_buffer,
                     Ray* double_buffer,
                     int* activePaths,
                     LocalGeometry* geom_buffer,
                     bool useRussianRoulette)
        {
            uint32_t tid = ThreadHelper::globalThreadIndex();

            if(tid >= numPaths)
            {
                return;
            }

            Ray& ray = ray_buffer[tid];
            LocalGeometry& geom = geom_buffer[tid];

            Vector3float inc_dir = -1.0f * ray.direction(); //Points away from surface
            if (geom.depth == INFINITY)
            {
                if (scene.useEnvironmentMap)
                {
                    Vector2uint32_t pixel = Tracing::direction2UV(ray.direction(), scene.environment.width(), scene.environment.height());
                    ray.payload<RadiancePayload>()->radiance += ray.payload<RadiancePayload>()->rayweight * scene.environment(pixel);
                }
                return;
            }

            RadiancePayload* payload = ray.payload<RadiancePayload>();
            directIllumination(scene, ray, geom, inc_dir);
            indirectIllumination(ray, geom, inc_dir);
            if(!payload->next_ray_valid) return;

            if(useRussianRoulette)
            {
                float alpha = Math::clamp(fmaxf(payload->rayweight.x, fmaxf(payload->rayweight.y, payload->rayweight.z)), 0.0f, 1.0f);
                if(Math::rnd(payload->seed) > alpha || Math::safeFloatEqual(alpha, 0))
                {
                    payload->next_ray_valid = false;
                    payload->rayweight = 0;
                    return;
                }
                payload->rayweight = payload->rayweight / alpha;
            }


            int work_idx = atomicAdd(activePaths, 1);

            ray.traceNew(payload->ray_start + 0.001f * payload->out_dir, payload->out_dir);

            double_buffer[work_idx] = ray;
        }

        CUPBR_GLOBAL void
        compose_kernel(RadiancePayload* payloads,
                       const uint32_t numPaths,
                       const uint32_t frameIndex,
                       Image<Vector3float> img)
        {
            uint32_t tid = ThreadHelper::globalThreadIndex();

            if(tid >= numPaths)
            {
                return;
            }

            RadiancePayload& payload = payloads[tid];

            if (frameIndex > 0)
            {
                const float a = 1.0f / (static_cast<float>(frameIndex) + 1.0f);
                payload.radiance = (1.0f - a) * img[tid] + a * payload.radiance;
            }

            img[tid] = payload.radiance;
        }

    } //namespace detail
    
    class RendererWavefront : public RenderMethod
    {
        public:

        RendererWavefront(Properties* properties)
        {
            max_trace_depth = properties->getProperty("max_trace_depth", 5);
            use_russian_roulette = properties->getProperty("use_russian_roulette", true);
            width = 1;
            height = 1;

            cudaSafeCall(cudaMalloc((void**)&ray_buffer, sizeof(Ray) * width * height));
            cudaSafeCall(cudaMalloc((void**)&double_buffer, sizeof(Ray) * width * height));
            cudaSafeCall(cudaMalloc((void**)&geom_buffer, sizeof(LocalGeometry) * width * height));
            cudaSafeCall(cudaMalloc((void**)&payload_buffer, sizeof(detail::RadiancePayload) * width * height));
            cudaSafeCall(cudaMalloc((void**)&activePaths, sizeof(int)));
        }

        ~RendererWavefront()
        {
            cudaSafeCall(cudaFree(ray_buffer));
            cudaSafeCall(cudaFree(double_buffer));
            cudaSafeCall(cudaFree(geom_buffer));
            cudaSafeCall(cudaFree(payload_buffer));
            cudaSafeCall(cudaFree(activePaths));
        }

        virtual void 
        render(Scene& scene,
               const Camera& camera,
               const uint32_t& frameIndex,
               Image<Vector3float>* output_img) 
        {
            /*const KernelSizeHelper::KernelSize config = KernelSizeHelper::configure(output_img->size());
            detail::volume_kernel << <config.blocks, config.threads >> > (scene,
                                                                          camera,
                                                                          frameIndex,
                                                                          max_trace_depth,
                                                                          use_russian_roulette,
                                                                          *output_img);
            synchronizeDefaultStream();*/

            if(output_img->width() != width || output_img->height() != height)
            {
                width = output_img->width();
                height = output_img->height();
                
                cudaSafeCall(cudaFree(ray_buffer));
                cudaSafeCall(cudaFree(geom_buffer));
                cudaSafeCall(cudaFree(payload_buffer));
                cudaSafeCall(cudaFree(double_buffer));

                cudaSafeCall(cudaMalloc((void**)&ray_buffer, sizeof(Ray) * width * height));
                cudaSafeCall(cudaMalloc((void**)&double_buffer, sizeof(Ray) * width * height));
                cudaSafeCall(cudaMalloc((void**)&geom_buffer, sizeof(LocalGeometry) * width * height));
                cudaSafeCall(cudaMalloc((void**)&payload_buffer, sizeof(detail::RadiancePayload) * width * height));

            }

            int numPaths = width * height;

            KernelSizeHelper::KernelSize config = KernelSizeHelper::configure(numPaths);
            detail::generate_kernel<<<config.blocks, config.threads>>>( camera,
                                                                        frameIndex,
                                                                        numPaths,
                                                                        *output_img,
                                                                        ray_buffer,
                                                                        payload_buffer);
            synchronizeDefaultStream();

            for(int i = 0; i < max_trace_depth; ++i)
            {
                KernelSizeHelper::KernelSize config1 = KernelSizeHelper::configure(numPaths);
                detail::extend_kernel<<<config1.blocks, config1.threads>>>(   scene,
                    numPaths,
                    ray_buffer,
                    geom_buffer,
                    activePaths);
                synchronizeDefaultStream();

                detail::shade_kernel<<<config1.blocks, config1.threads>>>(scene,
                                numPaths,
                                ray_buffer,
                                double_buffer,
                                activePaths,
                                geom_buffer,
                                use_russian_roulette);
                synchronizeDefaultStream();

                cudaSafeCall(cudaMemcpy((void*)&numPaths, (void*)activePaths, sizeof(int), cudaMemcpyDeviceToHost));

                if(numPaths == 0)
                    break;

                std::swap(ray_buffer, double_buffer);
            }

            KernelSizeHelper::KernelSize config2 = KernelSizeHelper::configure(width * height);
            detail::compose_kernel<<<config2.blocks, config2.threads>>>(payload_buffer,
                                                                        width * height,
                                                                        frameIndex,
                                                                        *output_img);
            synchronizeDefaultStream();
        }
        
        private:
        uint32_t max_trace_depth;
        bool use_russian_roulette;
        uint32_t width, height;
        Ray* ray_buffer, *double_buffer;
        LocalGeometry* geom_buffer;
        detail::RadiancePayload* payload_buffer;
        int* activePaths;
    };

    DEFINE_PLUGIN(RendererWavefront, "WavefrontRenderer", "1.0", RenderMethod)

}