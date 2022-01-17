#include <Renderer/SphereTracer.h>
#include <Core/KernelHelper.h>
#include <Core/Tracing.h>
#include <Geometry/Sphere.h>
#include <Geometry/Plane.h>
#include <Math/Matrix.h>

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

        __device__ bool generateFullVariablesWithModel(Ray& ray,
                                                       cunet::LenGen& lenGen,
                                                       cunet::PathGen& pathGen,
                                                       cunet::ScatGen& scatGen,
                                                       const float& g, 
                                                       const float& phi,
                                                       const Vector3float& win,
                                                       const float& density,
                                                       Vector3float& x,
                                                       Vector3float& w,
                                                       Vector3float& X,
                                                       Vector3float& W,
                                                       float& factor)
        {
            RadiancePayload* payload = ray.payload<RadiancePayload>();

            x = 0;
            w = win;
            X = x;
            W = w;
            factor = 1;

            Vector3float temp = fabsf(win.x) >= 0.9999 ? Vector3float(0, 0, 1) : Vector3float(1, 0, 0);
            Vector3float winY = Math::normalize(Math::cross(temp, win));
            Vector3float winX = Math::cross(win, winY);
            float rAlpha = Math::rnd(payload->seed) * 2 * static_cast<float>(M_PI);
            Matrix3x3float local(winX, winY, win);
            Matrix3x3float rot(Vector3float(cosf(rAlpha), sinf(rAlpha), 0), Vector3float(-sinf(rAlpha), cosf(rAlpha), 0), Vector3float(0, 0, 1));
            Matrix3x3float R = rot * local;

            float codedDensity = density;
            
            Vector2float lenLatent = Math::sampleStdNormal2D(payload->seed);
            float lenInput[4] = { codedDensity, g, lenLatent.x, lenLatent.y };
            float lenOutput[2];
            lenGen(lenInput, lenOutput); //TODO: This doesn't work because of race conditions

            float logN = fmaxf(0.0f, lenOutput[0] + Math::sampleStdNormal1D(payload->seed) * expf(Math::clamp(lenOutput[1],-16.0f,16.0f)));
            float n = roundf(expf(logN) + 0.49);
            logN = logf(n);

            float prob = 1.0f;
            for (uint32_t i = 0; i < n; ++i)    //pow breaks with cuda fast math
                prob *= phi;

            if (Math::rnd(payload->seed) >= prob)
                return false;

            Vector4float pathLatent14 = Math::sampleStdNormal4D(payload->seed);
            float pathLatent5 = Math::sampleStdNormal1D(payload->seed);
            float pathInput[8] = { codedDensity, g, logN, pathLatent14.x, pathLatent14.y, pathLatent14.z, pathLatent14.w, pathLatent5 };
            float pathOutput[6];
            pathGen(pathInput, pathOutput);
            
            Vector3float sampling = Math::sampleStdNormal3D(payload->seed);
            Vector3float pathMu = Vector3float(pathOutput[0], pathOutput[1], pathOutput[2]);
            Vector3float pathLogVar = Vector3float(
                Math::clamp(pathOutput[3], -16.0f, 16.0f), Math::clamp(pathOutput[4], -16.0f, 16.0f), Math::clamp(pathOutput[5], -16.0f, 16.0f));
            Vector3float pathOut = pathMu + Math::exp(pathLogVar * 0.5f) * sampling;
            pathOut.x = Math::clamp(pathOut.x, -0.9999f, 0.9999f);
            pathOut.y = Math::clamp(pathOut.y, -0.9999f, 0.9999f);
            pathOut.z = Math::clamp(pathOut.z, -0.9999f, 0.9999f);
            float cosTheta = pathOut.x;
            float wt = n >= 2 ? pathOut.y : 0.0f;
            float wb = n >= 3 ? pathOut.z : 0.0f;
            x = Vector3float(0, sqrtf(1 - cosTheta * cosTheta), cosTheta);
            Vector3float N = x;
            Vector3float B = Vector3float(1, 0, 0);
            Vector3float T = Math::cross(x, B);

            w = Math::normalize(N * sqrtf(fmaxf(0.0f, 1.0f - wt * wt - wb * wb)) + T * wt + B * wb);
            x = R * x;
            w = R * w;

            Vector4float scatLatent14 = Math::sampleStdNormal4D(payload->seed);
            float scatLatent5 = Math::sampleStdNormal1D(payload->seed);
            float scatInput[12] =
            {
                codedDensity,
                g,
                pow(1.0 - phi, 1.0 / 6.0),
                logN,
                cosTheta,
                wt,
                wb,
                scatLatent14.x,
                scatLatent14.y,
                scatLatent14.z,
                scatLatent14.w,
                scatLatent5
            };
            float scatOutput[12];
            scatGen(scatInput, scatOutput);

            if(n >= 2)
            {
                Vector3float sample1 = Math::sampleStdNormal3D(payload->seed);
                Vector3float sample2 = Math::sampleStdNormal3D(payload->seed);
                X = Vector3float(scatOutput[0] + Math::sampleStdNormal1D(payload->seed) * expf(Math::clamp(scatOutput[6], -16.0f, 16.0f)),
                                 scatOutput[1] + Math::sampleStdNormal1D(payload->seed) * expf(Math::clamp(scatOutput[7], -16.0f, 16.0f)),
                                 scatOutput[2] + Math::sampleStdNormal1D(payload->seed) * expf(Math::clamp(scatOutput[8], -16.0f, 16.0f)));
                X /= fmaxf(1.0f, Math::norm(X));
                W = Vector3float(scatOutput[3] + Math::sampleStdNormal1D(payload->seed) * expf(Math::clamp(scatOutput[9], -16.0f, 16.0f)),
                                 scatOutput[4] + Math::sampleStdNormal1D(payload->seed) * expf(Math::clamp(scatOutput[10], -16.0f, 16.0f)),
                                 scatOutput[5] + Math::sampleStdNormal1D(payload->seed) * expf(Math::clamp(scatOutput[11], -16.0f, 16.0f)));

                X = R * X;
                W = R * W;

                float accum = phi >= 0.99999 ? n : phi * (1.0f - prob) / (1.0f - phi);
                factor = accum / prob;
            }

            return true;
        }

        __device__ void directIlluminationST(Scene& scene, Ray& ray, LocalGeometry& geom, Vector3float& inc_dir)
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
                    Math::exp(-1.0f*(payload->volume->sigma_s + payload->volume->sigma_a) * fminf(d, 100000000.0f)) *
                    geom.material.brdf(geom.P, inc_dir, lightDir, normal) *
                    lightRadiance *
                    payload->rayweight;
            }
        }

        __device__ void indirectIlluminationST(Ray& ray, LocalGeometry& geom, Vector3float& inc_dir)
        {
            //Indirect illumination
            RadiancePayload* payload = ray.payload<RadiancePayload>();
            Vector4float direction_p = geom.material.sampleDirection(payload->seed, inc_dir, geom.N);
            Vector3float direction = Vector3float(direction_p);
            if (Math::norm(direction) == 0)
                return;
            ray.payload<RadiancePayload>()->rayweight = ray.payload<RadiancePayload>()->rayweight *
                fabs(Math::dot(direction, geom.N)) *
                geom.material.brdf(geom.P, inc_dir, direction, geom.N) / direction_p.w;
            payload->out_dir = direction;
            payload->ray_start = geom.P;
            payload->next_ray_valid = true;
        }

        __device__ bool handleMediumInteractionST(Scene& scene, Ray& ray, LocalGeometry& geom, Vector3float& inc_dir)
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

                bool usePT = true;

                if(usePT)
                {
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
                        Vector4float sample = geom.material.sampleDirection(payload->seed, inc_dir, geom.N);
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
                        attenuation = Math::exp(-1.0f * sigma_t * ge.depth);
                        shadow_ray.traceNew(ge.P + 0.001f * lightDir, lightDir);
                        d -= ge.depth;
                    }
                    else
                    {
                        shadow_ray = Ray(event_position + 0.001f * lightDir, lightDir);
                    }

                    Vector3float scene_sigma_t = scene.volume.sigma_a + scene.volume.sigma_s;
                    float scene_g = scene.volume.g;

                    if (Tracing::traceVisibility(scene, d, shadow_ray))
                    {
                        payload->radiance += (float)(scene.light_count + useEnvironmentMap) *
                            Math::exp(-1.0f*scene_sigma_t * fminf(d, 100000000.0f)) * attenuation *
                            lightRadiance *
                            payload->rayweight;
                        //Phase/pdf = 1
                    }

                    //Indirect Illumination
                    Vector4float sample_hg = sampleHenyeyGreensteinPhaseFunction(g, inc_dir, payload->seed);
                    payload->out_dir = Vector3float(sample_hg);
                    payload->ray_start = event_position;
                    //Phase/pdf = 1

                    payload->next_ray_valid = true;
                    return true;
                }
                else
                {
                    //SPHERE TRACING
                    //TODO: This is hard coded for now to test on the simple sphere scene
                    Sphere* sphere = static_cast<Sphere*>(scene[0]);
                    float r = sphere->radius() - Math::norm(event_position - sphere->position());
                    float er = payload->volume->sigma_a[channel] * r;


                }
                
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

        __global__ void
        sphere_kernel(Scene scene,
                      const Camera camera,
                      const uint32_t frameIndex,
                      const uint32_t maxTraceDepth,
                      const bool useRussianRoulette,
                      cunet::LenGen lenGen,
                      cunet::PathGen pathGen,
                      cunet::ScatGen scatGen,
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

                if (!handleMediumInteractionST(scene, ray, geom, inc_dir))
                {

                    //Handle medium interfaces
                    if(geom.material.type == MaterialType::VOLUME)
                    {
                        payload.out_dir = geom.material.sampleDirection(payload.seed, inc_dir, geom.N);
                        payload.ray_start = geom.P;
                        bool reflect = Math::dot(inc_dir, geom.N) * Math::dot(payload.out_dir, geom.N) > 0;
                        payload.inside_object = reflect ? payload.inside_object : !payload.inside_object;
                        payload.object_index = geom.scene_index;
                        payload.next_ray_valid = true;
                        payload.volume = payload.inside_object ? &(geom.material.volume) : &(scene.volume);
                    }
                    else
                    {
                        directIlluminationST(scene, ray, geom, inc_dir);
                        indirectIlluminationST(ray, geom, inc_dir);
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

    void
    PBRendering::spheretracing(Scene& scene,
                               const Camera& camera,
                               const uint32_t& frameIndex,
                               const uint32_t& maxTraceDepth,
                               const bool& useRussianRoulette,
                               cunet::LenGen& lenGen,
                               cunet::PathGen& pathGen,
                               cunet::ScatGen& scatGen,
                               Image<Vector3float>* output_img)
    {
        const KernelSizeHelper::KernelSize config = KernelSizeHelper::configure(output_img->size());
        detail::sphere_kernel << <config.blocks, config.threads >> > (scene,
                                                                      camera,
                                                                      frameIndex,
                                                                      maxTraceDepth,
                                                                      useRussianRoulette,
                                                                      lenGen,
                                                                      pathGen,
                                                                      scatGen,
                                                                      *output_img);
        cudaSafeCall(cudaDeviceSynchronize());
    }

} //namespace cupbr
