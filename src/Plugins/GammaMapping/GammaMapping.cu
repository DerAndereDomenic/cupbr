#include <CUPBR.h>

namespace cupbr
{

    namespace detail
    {
        CUPBR_DEVICE
        float
        apply_srgb_gamma(const float& c, const float& gamma)
        {
            return c <= 0.0031308f ? 12.92f * c : 1.055f * powf(c, 1.0f / gamma) - 0.055f;
        }

        CUPBR_GLOBAL
        void
        gamma_kernel(Image<Vector3float> hdr_image, RenderBuffer output, const float exposure, const float gamma)
        {
            const uint32_t tid = ThreadHelper::globalThreadIndex();

            if (tid >= hdr_image.size())
            {
                return;
            }

            Vector3float radiance = hdr_image[tid];

            output[tid] = Vector4uint8_t(
                static_cast<uint8_t>(Math::clamp(apply_srgb_gamma(exposure * radiance.x, gamma), 0.0f, 1.0f) * 255.0f),
                static_cast<uint8_t>(Math::clamp(apply_srgb_gamma(exposure * radiance.y, gamma), 0.0f, 1.0f) * 255.0f),
                static_cast<uint8_t>(Math::clamp(apply_srgb_gamma(exposure * radiance.z, gamma), 0.0f, 1.0f) * 255.0f),
                255u
            );
        }
    } //namespace detail
    
    class GammaMapping : public ToneMappingMethod
    {
        public:

        GammaMapping(Properties* properties)
        {
            exposure = properties->getProperty("exposure", 1.0f);
            gamma = properties->getProperty("gamma", 2.4f);
        }

        virtual void 
        toneMap(Image<Vector3float>& hdr_image,
                RenderBuffer& output_img)
        {
            KernelSizeHelper::KernelSize config = KernelSizeHelper::configure(hdr_image.size());
            detail::gamma_kernel << <config.blocks, config.threads >> > (hdr_image, output_img, exposure, gamma);
            synchronizeDefaultStream();
        }
        
        private:
        float exposure;
        float gamma;
    };

    DEFINE_PLUGIN(GammaMapping, "GammaMapping", "1.0", ToneMappingMethod)

}