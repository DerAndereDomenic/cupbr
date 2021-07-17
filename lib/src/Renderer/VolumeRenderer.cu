#include <Renderer/VolumeRenderer.cuh>
#include <Core/KernelHelper.cuh>
#include <Core/Tracing.cuh>
#include <Geometry/Sphere.cuh>
#include <Geometry/Plane.cuh>

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

    __device__ float henyeyGreensteinPhaseFunction(const float& g, const float& cos_theta)
    {
        float g2 = g * g;
        float area = 4.0f * 3.14159f;
        return (1 - g2) / area * powf((1 + g2 - 2.0f * g * cos_theta), -1.5f);
    }

    __device__ Vector4float sampleHenyeyGreensteinPhaseFunction(const float& g, const Vector3float& forward, uint32_t& seed)
    {
        float u1 = Math::rnd(seed);
        float u2 = Math::rnd(seed);
        
        float g2 = g * g;
        float d = (1.0f - g2) / (1.0f - g + 2.0f*g*u1);
        float cos_theta = 0.5f/g * (1.0f + g2 - d*d);

        float sin_theta = sqrtf(fmaxf(0.0f, 1.0f-cos_theta*cos_theta));
        float phi = 2.0f * 3.14159f * u2;

        float x = sin_theta * cosf(phi);
        float y = sin_theta * sinf(phi);
        float z = cos_theta;

        Vector3float result = Math::normalize(Math::toLocalFrame(forward, Vector3float(x,y,z)));

        float pdf = henyeyGreensteinPhaseFunction(g, Math::dot(forward, result));

        return Vector4float(result, pdf);
    }

    __device__ void directIlluminationVolumetric(Scene& scene, Ray& ray, LocalGeometry& geom, Vector3float& inc_dir)
    {
        //Direct illumination
        RadiancePayload* payload = ray.payload<RadiancePayload>();

        Vector3float normal = geom.N;

        //Don't shade back facing geometry
        if(geom.material.type != GLASS && Math::dot(normal, inc_dir) <= 0.0f)
        {
            payload->rayweight = 0;
            return;
        }

        uint32_t useEnvironmentMap = scene.useEnvironmentMap ? 1 : 0;
        uint32_t light_sample = static_cast<uint32_t>(Math::rnd(payload->seed) * (scene.light_count + useEnvironmentMap));

        Light light;
        Vector3float lightDir, lightRadiance;
        float d;
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
                    float xi1 = Math::rnd(payload->seed) * 2.0f - 1.0f;
                    float xi2 = Math::rnd(payload->seed) * 2.0f - 1.0f;

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
            Vector4float sample = geom.material.sampleDirection(payload->seed, inc_dir, geom.N);
            lightDir = Vector3float(sample);
            d = INFINITY; //TODO: Better way to do this
            Vector2uint32_t pixel = Tracing::direction2UV(lightDir, scene.environment.width(), scene.environment.height());
            lightRadiance = scene.environment(pixel)/sample.w;
        }
            
        Ray shadow_ray = Ray(geom.P + 0.001f*lightDir, lightDir);

        if(Tracing::traceVisibility(scene, d, shadow_ray))
        {
            payload->radiance += (scene.light_count+useEnvironmentMap) *
                                                        fmaxf(0.0f, Math::dot(normal,lightDir)) *
                                                        expf(-1.1f * d) *
                                                        geom.material.brdf(geom.P,inc_dir,lightDir,normal) *
                                                        lightRadiance *
                                                        payload->rayweight;
        }
    }

    __device__ void indirectIlluminationVolumetric(Ray& ray, LocalGeometry& geom, Vector3float& inc_dir)
    {
        //Indirect illumination
        RadiancePayload* payload = ray.payload<RadiancePayload>();
        Vector4float direction_p = geom.material.sampleDirection(payload->seed, inc_dir, geom.N);
        Vector3float direction = Vector3float(direction_p);
        if (Math::norm(direction) == 0)
            return;
        ray.payload<RadiancePayload>()->rayweight = ray.payload<RadiancePayload>()->rayweight * 
                                                    fabs(Math::dot(direction, geom.N)) * 
                                                    geom.material.brdf(geom.P, inc_dir, direction, geom.N)/direction_p.w;
        payload->out_dir = direction;
        payload->ray_start = geom.P;
        payload->next_ray_valid = true;
    }

    __device__ bool handleMediumInteraction(Scene& scene, Ray& ray, LocalGeometry& geom, Vector3float& inc_dir)
    {
        RadiancePayload* payload = ray.payload<RadiancePayload>();

        float g = 0.0f;
        float sigma_a = 0.1f;
        float sigma_s = 1.0f;
        float sigma_t = sigma_a + sigma_s;

        float t = -1.0f/sigma_t * logf(Math::rnd(payload->seed));

        if(t <= geom.depth)
        {
            Vector3float event_position = ray.origin() + t*ray.direction();

            float scattering_prob = sigma_s / sigma_t;

            if(Math::rnd(payload->seed) < scattering_prob)
            {
                //Attenuate ray from its start to the medium event
                payload->rayweight = payload->rayweight * 
                                     sigma_s / scattering_prob *
                                     expf(-sigma_t * t) / (sigma_t * expf(-sigma_t * t));
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
            if(light_sample != scene.light_count)
            {
                light = *(scene.lights[light_sample]); 

                switch(light.type)
                {
                    case LightType::POINT:
                    {
                        lightDir = Math::normalize(light.position - event_position);
                        d = Math::norm(light.position - event_position);
                        lightRadiance = light.intensity / (d*d);
                    }
                    break;
                    case LightType::AREA:
                    {
                        float xi1 = Math::rnd(payload->seed) * 2.0f - 1.0f;
                        float xi2 = Math::rnd(payload->seed) * 2.0f - 1.0f;

                        Vector3float sample = light.position + xi1 * light.halfExtend1 + xi2 * light.halfExtend2;
                        Vector3float n = Math::normalize(Math::cross(light.halfExtend1, light.halfExtend2));
                        float area = 4.0f*Math::norm(light.halfExtend1) * Math::norm(light.halfExtend2);

                        lightDir = Math::normalize(sample - event_position);
                        d = Math::norm(sample - event_position);

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
                Vector4float sample = geom.material.sampleDirection(payload->seed, inc_dir, geom.N);
                lightDir = Vector3float(sample);
                d = INFINITY; //TODO: Better way to do this
                Vector2uint32_t pixel = Tracing::direction2UV(lightDir, scene.environment.width(), scene.environment.height());
                lightRadiance = scene.environment(pixel)/sample.w;
            }
                
            Ray shadow_ray = Ray(event_position + 0.001f*lightDir, lightDir);

            if(Tracing::traceVisibility(scene, d, shadow_ray))
            {
                payload->radiance += (scene.light_count+useEnvironmentMap) *
                                      henyeyGreensteinPhaseFunction(g, Math::dot(lightDir, inc_dir)) *
                                      expf(-sigma_t * d) *
                                      lightRadiance *
                                      payload->rayweight;
            }

            //Indirect Illumination
            payload->out_dir = Vector3float(sampleHenyeyGreensteinPhaseFunction(g, inc_dir, payload->seed));
            payload->ray_start = event_position;

            payload->next_ray_valid = true;
            return true;
        }
        else
        {
            float pdf = expf(-sigma_t * geom.depth);
            payload->rayweight = payload->rayweight * expf(-sigma_t * geom.depth) / pdf;
            payload->next_ray_valid = true;
            payload->ray_start = geom.P;
            return false;
        }
    }

    __global__ void
    volume_kernel(Scene scene,
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
        payload.seed = seed;
        ray.setPayload(&payload);

        uint32_t trace_depth = 0;
        Vector3float inc_dir;

        Light light;

        do
        {
            payload.next_ray_valid = false;
            LocalGeometry geom = Tracing::traceRay(scene, ray);
            if(geom.depth == INFINITY)
            {
                if(scene.useEnvironmentMap)
                {
                    Vector2uint32_t pixel = Tracing::direction2UV(ray.direction(), scene.environment.width(), scene.environment.height());
                    payload.radiance += payload.rayweight * scene.environment(pixel);
                }
                break;
            }
            Vector3float inc_dir = Math::normalize(ray.origin() - geom.P);

            if(!handleMediumInteraction(scene, ray, geom, inc_dir))
            {
                directIlluminationVolumetric(scene, ray, geom, inc_dir);
                indirectIlluminationVolumetric(ray, geom, inc_dir);
            }
                 
            ray.traceNew(payload.ray_start+0.01f*payload.out_dir, payload.out_dir);

            if (!payload.next_ray_valid)break;
            ++trace_depth;
        }while(trace_depth < maxTraceDepth);

        if(frameIndex > 0)
        {
            const float a = 1.0f/(static_cast<float>(frameIndex) + 1.0f);
            ray.payload<RadiancePayload>()->radiance = (1.0f-a)*img[tid] + a*ray.payload<RadiancePayload>()->radiance;
        }

        img[tid] = ray.payload<RadiancePayload>()->radiance;
    }
}

void
PBRendering::volumetracing(Scene& scene,
                           const Camera& camera,
                           const uint32_t& frameIndex,
                           const uint32_t& maxTraceDepth,
                           Image<Vector3float>* output_img)
{
    const KernelSizeHelper::KernelSize config = KernelSizeHelper::configure(output_img->size());
    detail::volume_kernel<<<config.blocks, config.threads>>>(scene, 
                                                             camera,
                                                             frameIndex,
                                                             maxTraceDepth, 
                                                             *output_img);
    cudaSafeCall(cudaDeviceSynchronize());
}
