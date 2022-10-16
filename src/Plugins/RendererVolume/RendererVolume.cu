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
            Volume* volume;
            bool inside_object = false;
            uint32_t object_index = 0;
        };

        CUPBR_DEVICE void directIlluminationVolumetric(GeometryScene& scene, Ray& ray, LocalGeometry& geom, Vector3float& inc_dir)
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
                    Math::exp(-1.0f*(payload->volume->sigma_s + payload->volume->sigma_a) * fminf(d, 100000000.0f)) *
                    geom.material->brdf(geom.P, inc_dir, lightDir, normal) *
                    lightRadiance *
                    payload->rayweight;
            }
        }

        CUPBR_DEVICE void indirectIlluminationVolumetric(Ray& ray, LocalGeometry& geom, Vector3float& inc_dir)
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

        CUPBR_DEVICE bool handleMediumInteraction(GeometryScene& scene, Ray& ray, LocalGeometry& geom, Vector3float& inc_dir)
        {
            RadiancePayload* payload = ray.payload<RadiancePayload>();

            float g = payload->volume->g;
            Vector3float sigma_a = payload->volume->sigma_a;
            Vector3float sigma_s = payload->volume->sigma_s;
            Vector3float sigma_t = sigma_a + sigma_s;

            uint32_t channel = static_cast<uint32_t>(Math::rnd(payload->seed) * 3);

            if (Math::safeFloatEqual(sigma_t[channel], 0.0f))
                return false;

            float t =  - logf(1.0f - Math::rnd(payload->seed)) / sigma_t[channel];

            if (t < geom.depth)
            {
                Vector3float event_position = ray.origin() + t * ray.direction();
                float scattering_prob = sigma_s[channel] / sigma_t[channel];

                if (Math::rnd(payload->seed) < scattering_prob)
                {
                    //Attenuate ray from its start to the medium event
                    //For monochrome sigma this is 1 but it may change for mulit channel scattering

                    float pdf = 0.0f;
                    for(uint32_t i = 0; i < 3; ++i)
                    {
                        pdf += sigma_t[i] * expf(-1.0f * sigma_t[i] * t);
                    }
                    pdf /= 3.0f;

                    payload->rayweight = payload->rayweight *
                        sigma_s / scattering_prob *
                        Math::exp(-1.0f*sigma_t * t) / pdf;
                }
                else
                {
                    payload->rayweight = 0;
                    payload->next_ray_valid = false;
                    return true;
                }

                //Direct illumination
                uint32_t useEnvironmentMap = scene.useEnvironmentMap ? 1 : 0;
                uint32_t light_sample = static_cast<uint32_t>(Math::rnd(payload->seed) * (scene.light_count + useEnvironmentMap));

                Light light;
                Vector3float lightDir, lightRadiance;
                float d;
                if (light_sample != scene.light_count)
                {
                    light = *(scene.lights[light_sample]);

                    lightRadiance = light.sample(payload->seed, event_position, lightDir, d);
                }
                else // Use environment map
                {
                    Vector4float sample = geom.material->sampleDirection(payload->seed, inc_dir, geom.N);
                    lightDir = Vector3float(sample);
                    d = INFINITY; //TODO: Better way to do this
                    Vector2uint32_t pixel = Tracing::direction2UV(lightDir, scene.environment.width(), scene.environment.height());
                    lightRadiance = scene.environment(pixel) / sample.w;
                }

                Ray shadow_ray;

                // If we are inside an object -> First move to border of current object -> then do light sampling
                //TODO: This is still not optimal for some scenarios because of numerical issues (?)
                Vector3float attenuation = 1.0f;
                if(payload->inside_object)
                {
                    shadow_ray = Ray(event_position, lightDir);
                    LocalGeometry ge = Tracing::traceRay(scene, shadow_ray, payload->object_index);
                    if (ge.depth == INFINITY) 
                    {
                        ge.depth = 0;
                        ge.P = event_position;
                    }
                    attenuation = Math::exp(-1.0f * sigma_t * ge.depth) * Material::henyeyGreensteinPhaseFunction(g, -Math::dot(inc_dir,lightDir));
                    shadow_ray.traceNew(ge.P + 0.001f * lightDir, lightDir);
                    d -= ge.depth;
                }
                else
                {
                    shadow_ray = Ray(event_position + 0.001f * lightDir, lightDir);
                }

                Vector3float scene_sigma_t = scene.volume.sigma_a + scene.volume.sigma_s;

                if (Tracing::traceVisibility(scene, d, shadow_ray))
                {
                    payload->radiance += (float)(scene.light_count + useEnvironmentMap) *
                        Math::exp(-1.0f*scene_sigma_t * fminf(d, 100000000.0f)) * attenuation *
                        lightRadiance *
                        payload->rayweight;
                    //Phase/pdf = 1
                }

                //Indirect Illumination
                Vector4float sample_hg = Material::sampleHenyeyGreensteinPhaseFunction(g, -1.0f * inc_dir, payload->seed);
                payload->out_dir = Vector3float(sample_hg);
                payload->ray_start = event_position;
                //Phase/pdf = 1

                payload->next_ray_valid = true;
                return true;
            }
            else
            {
                float pdf = 0.0f;

                for(uint32_t i = 0; i < 3; ++i)
                {
                    pdf += expf(-sigma_t[i] * geom.depth);
                }
                pdf /= 3.0f;

                payload->rayweight = payload->rayweight * Math::exp(-1.0f*sigma_t * geom.depth) / pdf;
                payload->ray_start = geom.P;
                return false;
            }
        }

        CUPBR_GLOBAL void
        volume_kernel(GeometryScene scene,
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
            payload.volume = &(scene.volume);
            ray.setPayload(&payload);

            uint32_t trace_depth = 0;

            Light light;

            do
            {
                payload.next_ray_valid = false;
                LocalGeometry geom;
                    
                if(!payload.inside_object)
                {
                    geom = Tracing::traceRay(scene, ray);
                }
                else
                {
                    geom = Tracing::traceRay(scene, ray, payload.object_index);
                }

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

                if (!handleMediumInteraction(scene, ray, geom, inc_dir))
                {
                    //Handle medium interfaces
                    if(geom.material->type == MaterialType::REFRACTIVE)
                    {
                        payload.out_dir = geom.material->sampleDirection(payload.seed, inc_dir, geom.N);
                        payload.ray_start = geom.P;
                        bool reflect = Math::dot(inc_dir, geom.N) * Math::dot(payload.out_dir, geom.N) > 0;
                        payload.inside_object = reflect ? payload.inside_object : !payload.inside_object;
                        payload.object_index = geom.scene_index;
                        payload.next_ray_valid = true;
                        payload.volume = payload.inside_object ? &(geom.material->volume) : &(scene.volume);
                    }
                    else
                    {
                        directIlluminationVolumetric(scene, ray, geom, inc_dir);
                        indirectIlluminationVolumetric(ray, geom, inc_dir);
                    }

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
    } //namespace detail
    
    class RendererVolume : public RenderMethod
    {
        public:

        RendererVolume(Properties* properties)
        {
            max_trace_depth = properties->getProperty("max_trace_depth", 5);
            use_russian_roulette = properties->getProperty("use_russian_roulette", true);
        }

        virtual void 
        render(Scene* scene,
               const Camera& camera,
               const uint32_t& frameIndex,
               Image<Vector3float>* output_img) 
        {
            GeometryScene* geom_scene = dynamic_cast<GeometryScene*>(scene);

            if (geom_scene == nullptr)
            {
                std::cerr << "ERROR: VolumeRenderer received scene that does not hold geometry information!\n";
                return;
            }

            const KernelSizeHelper::KernelSize config = KernelSizeHelper::configure(output_img->size());
            detail::volume_kernel << <config.blocks, config.threads >> > (*geom_scene,
                                                                          camera,
                                                                          frameIndex,
                                                                          max_trace_depth,
                                                                          use_russian_roulette,
                                                                          *output_img);
            synchronizeDefaultStream();
        }
        
        private:
        uint32_t max_trace_depth;
        bool use_russian_roulette;
    };

    DEFINE_PLUGIN(RendererVolume, "VolumeRenderer", "1.0", RenderMethod)

}