#include <Renderer/GradientDomain.cuh>
#include <Core/KernelHelper.cuh>
#include <Core/Tracing.cuh>
#include <Geometry/Sphere.cuh>
#include <Geometry/Plane.cuh>

namespace detail
{
    #define DIFFUSE_THRESHOLD 0.1f

    struct PathData
    {
        Vector3float position;
        Vector3float normal;
        bool diffuse;
        
        float pdf;
        bool valid;
    };

    struct RadiancePayload
    {
        uint32_t seed;
        Vector3float radiance = 0;
        Vector3float rayweight = 1;
        Vector3float out_dir;
        bool next_ray_valid;
        uint32_t trace_depth = 0;

        PathData path[10]; //Hard code max trace depth
        PathData* base_path;
    };

    inline __device__
    void emissiveIllumintationGDPT(Ray& ray, LocalGeometry& geom)
    {
        RadiancePayload* payload = ray.payload<RadiancePayload>();

        payload->radiance += payload->rayweight * geom.material.albedo_e;
    }

    inline __device__ 
    void directIlluminationGDPT(Scene& scene, Ray& ray, LocalGeometry& geom, Vector3float& inc_dir)
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
                                                        geom.material.brdf(geom.P,inc_dir,lightDir,normal) *
                                                        lightRadiance *
                                                        payload->rayweight;
        }
    }

    inline __device__ 
    void indirectIlluminationGDPT(Ray& ray, LocalGeometry& geom, Vector3float& inc_dir)
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
        payload->next_ray_valid = true;
        payload->path[payload->trace_depth].pdf = direction_p.w;
    }

    __device__ void
    collect_radiance(Ray& ray,
                     Scene& scene,
                     const Camera& camera,
                     const uint32_t& maxTraceDepth)
    {
        RadiancePayload* payload = ray.payload<RadiancePayload>();

        uint32_t trace_depth = 0;
        Vector3float inc_dir;

        Light light;

        do
        {
            payload->next_ray_valid = false;
            LocalGeometry geom = Tracing::traceRay(scene, ray);
            if(geom.depth == INFINITY)
            {
                payload->path[payload->trace_depth].valid = false;
                if(scene.useEnvironmentMap)
                {
                    Vector2uint32_t pixel = Tracing::direction2UV(ray.direction(), scene.environment.width(), scene.environment.height());
                    payload->radiance += payload->rayweight * scene.environment(pixel);
                }
                break;
            }

            //Store path data
            payload->path[payload->trace_depth].position = geom.P;
            payload->path[payload->trace_depth].normal = geom.N;
            payload->path[payload->trace_depth].valid = true;
            payload->path[payload->trace_depth].diffuse = geom.material.shininess > DIFFUSE_THRESHOLD;

            Vector3float inc_dir = Math::normalize(ray.origin() - geom.P);

            emissiveIllumintationGDPT(ray, geom);
            directIlluminationGDPT(scene, ray, geom, inc_dir);
            indirectIlluminationGDPT(ray, geom, inc_dir);
                 
            ray.traceNew(geom.P+0.01f*payload->out_dir, payload->out_dir);

            if (!payload->next_ray_valid)break;
            ++trace_depth;
            ++payload->trace_depth;
        }while(trace_depth < maxTraceDepth);
    }

    __global__ void
    gdpt_kernel(Scene scene,
                Camera camera,
                const uint32_t frameIndex,
                const uint32_t maxTraceDepth,
                Image<Vector3float> img)
    {
        const uint32_t tid = ThreadHelper::globalThreadIndex();

        if(tid >= img.size())
        {
            return;
        }

        const Vector2int32_t pixel = ThreadHelper::index2pixel(tid, img.width(), img.height());
        if(pixel.x == 0 || pixel.x == img.width() - 1 || pixel.y == 0 || pixel.y == img.height() - 1)
        {
            return;
        }

        uint32_t seed = Math::tea<4>(tid, frameIndex);

        Ray base_ray = Tracing::launchRay(pixel, img.width(), img.height(), camera, true, &seed);
        RadiancePayload payload_base;
        payload_base.seed = seed;
        base_ray.setPayload(&payload_base);

        Ray left_ray = Tracing::launchRay(pixel + Vector2int32_t(-1,0), img.width(), img.height(), camera, true, &seed);
        RadiancePayload payload_left;
        payload_left.seed = seed;
        payload_left.base_path = payload_base.path;
        left_ray.setPayload(&payload_left);

        Ray right_ray = Tracing::launchRay(pixel + Vector2int32_t(1,0), img.width(), img.height(), camera, true, &seed);
        RadiancePayload payload_right;
        payload_right.seed = seed;
        payload_right.base_path = payload_base.path;
        right_ray.setPayload(&payload_right);

        Ray up_ray = Tracing::launchRay(pixel + Vector2int32_t(0,1), img.width(), img.height(), camera, true, &seed);
        RadiancePayload payload_up;
        payload_up.seed = seed;
        payload_up.base_path = payload_base.path;
        up_ray.setPayload(&payload_up);

        Ray down_ray = Tracing::launchRay(pixel + Vector2int32_t(0,-1), img.width(), img.height(), camera, true, &seed);
        RadiancePayload payload_down;
        payload_down.seed = seed;
        payload_down.base_path = payload_base.path;
        down_ray.setPayload(&payload_down);

        collect_radiance(base_ray, scene, camera, maxTraceDepth);
        collect_radiance(left_ray, scene, camera, maxTraceDepth);
        collect_radiance(right_ray, scene, camera, maxTraceDepth);
        collect_radiance(up_ray, scene, camera, maxTraceDepth);
        collect_radiance(down_ray, scene, camera, maxTraceDepth);

        Vector3float radiance = payload_base.radiance;

        if(frameIndex > 0)
        {
            const float a = 1.0f/(static_cast<float>(frameIndex) + 1.0f);
            radiance = (1.0f-a)*img[tid] + a*radiance;
        }

        img[tid] = radiance;
    }
}

void
PBRendering::gradientdomain(Scene& scene,
                            const Camera& camera,
                            const uint32_t& frameIndex,
                            const uint32_t& maxTraceDepth,
                            Image<Vector3float>* base,
                            Image<Vector3float>* temp,
                            Image<Vector3float>* gradient_x,
                            Image<Vector3float>* gradient_y,
                            Image<Vector3float>* output_img)
{
    const KernelSizeHelper::KernelSize config = KernelSizeHelper::configure(output_img->size());
    detail::gdpt_kernel<<<config.blocks, config.threads>>>(scene, 
                                                           camera,
                                                           frameIndex,
                                                           maxTraceDepth,
                                                           *output_img);
    cudaSafeCall(cudaDeviceSynchronize());
}
