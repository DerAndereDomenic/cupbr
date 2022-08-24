#include <CUPBR.h>

namespace cupbr
{

    namespace detail
    {
        CUPBR_GLOBAL void
        reinhard_kernel(Image<Vector3float> hdr_image, RenderBuffer output, const float exposure, const float gamma)
        {
            const uint32_t tid = ThreadHelper::globalThreadIndex();

            if (tid >= hdr_image.size())
            {
                return;
            }

            Vector3float radiance = hdr_image[tid];
            Vector3uint8_t color(0);

            float mapped_red = powf(1.0 - expf(-radiance.x * exposure), 1.0f / gamma);
            float mapped_green = powf(1.0 - expf(-radiance.y * exposure), 1.0f / gamma);
            float mapped_blue = powf(1.0 - expf(-radiance.z * exposure), 1.0f / gamma);

            uint8_t red = static_cast<uint8_t>(Math::clamp(mapped_red, 0.0f, 1.0f) * 255.0f);
            uint8_t green = static_cast<uint8_t>(Math::clamp(mapped_green, 0.0f, 1.0f) * 255.0f);
            uint8_t blue = static_cast<uint8_t>(Math::clamp(mapped_blue, 0.0f, 1.0f) * 255.0f);

            color = Vector3uint8_t(red, green, blue);

            output[tid] = Vector4uint8_t(color, 255);
        }
    } //namespace detail
    
    class ReinhardMapping : public ToneMappingMethod
    {
        public:

        ReinhardMapping(Properties* properties)
        {
            exposure = properties->getProperty("exposure", 1.0f);
            gamma = properties->getProperty("gamma", 2.4f);
        }

        virtual void 
        toneMap(Image<Vector3float>& hdr_image,
                RenderBuffer& output_img)
        {
            KernelSizeHelper::KernelSize config = KernelSizeHelper::configure(hdr_image.size());
            detail::reinhard_kernel << <config.blocks, config.threads >> > (hdr_image, output_img, exposure, gamma);
            synchronizeDefaultStream();
        }
        
        private:
        float exposure;
        float gamma;
    };

    DEFINE_PLUGIN(ReinhardMapping, "ReinhardMapping", "1.0", ToneMappingMethod)

}