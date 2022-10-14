#include <CUPBR.h>

namespace cupbr
{
    struct SDFPayload
    {
        bool hit = false;
        Vector3float pos;
    };

    struct SDF
    {
        CUPBR_DEVICE
        float operator()(const Vector3float& x)
        {
            return Math::norm(x - pos) - r;
        }

        Vector3float pos = Vector3float(0);
        float r = 1.0f;
    };

    namespace detail
    {
        CUPBR_GLOBAL void
        sdf_kernel(Scene scene,
                      const Camera camera,
                      const uint32_t frameIndex,
                      Image<Vector3float> img)
        {
            const uint32_t tid = ThreadHelper::globalThreadIndex();

            if (tid >= img.size())
            {
                return;
            }

            uint32_t seed = Math::tea<4>(tid, frameIndex);

            Ray ray = Tracing::launchRay(tid, img.width(), img.height(), camera, true, &seed);
            SDFPayload payload;

            SDF sdf;

            float total_distance = 0;
            float step_size = 0;
            do
            {
                float step_size = sdf(ray.origin());
                total_distance += step_size;

                if (step_size < 1e-5f)
                {
                    payload.hit = true;
                    payload.pos = ray.origin();
                    break;
                }

                ray.traceNew(ray.origin() + ray.direction() * step_size, ray.direction());

            } while (total_distance < 100);
            
            Vector3float radiance = 0;

            if(payload.hit)
            {
                float eps = 1e-5f;
                float diff_x = sdf(payload.pos + Vector3float(eps, 0, 0)) - sdf(payload.pos - Vector3float(eps, 0, 0));
                float diff_y = sdf(payload.pos + Vector3float(0, eps, 0)) - sdf(payload.pos - Vector3float(0, eps, 0));
                float diff_z = sdf(payload.pos + Vector3float(0, 0, eps)) - sdf(payload.pos - Vector3float(0, 0, eps));

                radiance = -Math::dot(Math::normalize(Vector3float(diff_x, diff_y, diff_z)), ray.direction());
            }

            if (frameIndex > 0)
            {
                const float a = 1.0f / (static_cast<float>(frameIndex) + 1.0f);
                radiance = (1.0f - a) * img[tid] + a * radiance;
            }

            img[tid] = radiance;
        }
    } //namespace detail
    
    class RendererSDF : public RenderMethod
    {
        public:

        RendererSDF(Properties* properties)
        {

        }

        virtual void 
        render(Scene* scene,
               const Camera& camera,
               const uint32_t& frameIndex,
               Image<Vector3float>* output_img) 
        {
            const KernelSizeHelper::KernelSize config = KernelSizeHelper::configure(output_img->size());
            detail::sdf_kernel << <config.blocks, config.threads >> > (*scene,
                                                                          camera,
                                                                          frameIndex,
                                                                          *output_img);
            synchronizeDefaultStream();
        }
    };

    DEFINE_PLUGIN(RendererSDF, "SDFRenderer", "1.0", RenderMethod)

}