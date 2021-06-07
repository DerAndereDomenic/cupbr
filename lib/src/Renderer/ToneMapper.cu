#include <Renderer/ToneMapper.cuh>

#include <DataStructure/Image.cuh>
#include <DataStructure/RenderBuffer.cuh>
#include <memory>

#include <Core/KernelHelper.cuh>
#include <Math/Functions.cuh>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace detail
{
    __global__ void
    reinhard_kernel(Image<Vector3float> hdr_image, RenderBuffer output)
    {
        const uint32_t tid = ThreadHelper::globalThreadIndex();

        if(tid >= hdr_image.size())
        {
            return;
        }

        Vector3float radiance = hdr_image[tid];
        Vector3uint8_t color(0);

        float mapped_red = powf(1.0 - expf(-radiance.x), 1.0f/2.2f);
        float mapped_green = powf(1.0 - expf(-radiance.y), 1.0f/2.2f);
        float mapped_blue = powf(1.0 - expf(-radiance.z), 1.0f/2.2);

        uint8_t red = static_cast<uint8_t>(Math::clamp(mapped_red, 0.0f, 1.0f)*255.0f);
        uint8_t green = static_cast<uint8_t>(Math::clamp(mapped_green, 0.0f, 1.0f)*255.0f);
        uint8_t blue = static_cast<uint8_t>(Math::clamp(mapped_blue, 0.0f, 1.0f)*255.0f);

        color = Vector3uint8_t(red, green, blue);

        output[tid] = Vector4uint8_t(color,255);
    }

    __device__
    float
    apply_srgb_gamma(const float& c)
    {
        return c <= 0.0031308f ? 12.92f * c : 1.055f * powf(c, 1.0f/2.4f) - 0.055f;
    }

    __global__
    void
    gamma_kernel(Image<Vector3float> hdr_image, RenderBuffer output)
    {
        const uint32_t tid = ThreadHelper::globalThreadIndex();

        if(tid >= hdr_image.size())
        {
            return;
        }

        Vector3float radiance = hdr_image[tid];

        output[tid] = Vector4uint8_t(
            static_cast<uint8_t>( Math::clamp( apply_srgb_gamma(radiance.x), 0.0f, 1.0f)*255.0f),
            static_cast<uint8_t>( Math::clamp( apply_srgb_gamma(radiance.y), 0.0f, 1.0f)*255.0f),
            static_cast<uint8_t>( Math::clamp( apply_srgb_gamma(radiance.z), 0.0f, 1.0f)*255.0f),
            255u
        );
    }
}

class ToneMapper::Impl
{
    public:
        Impl();

        ~Impl();

        /**
        *   @brief The Reinhard tone mapping algorithm
        */
        void
        toneMappingReinhard();

        /**
        *   @brief The Gamme tone mapping algorihm
        */
        void
        toneMappingGamma();

        //Data
        ToneMappingType type;               /**< The tone mapping algorithm */
        bool isRegistered;                  /**< If a HDR image has been registered */
        RenderBuffer render_buffer;         /**< The output render buffer */
        Image<Vector3float>* hdr_image;     /**< The registered HDR image */
};

ToneMapper::Impl::Impl()
{
    isRegistered = false;
}

ToneMapper::Impl::~Impl()
{
    if(isRegistered)
    {
        RenderBuffer::destroyDeviceObject(render_buffer);
    }
    isRegistered = false;
}

ToneMapper::ToneMapper(const ToneMappingType& type)
{
    impl = std::make_unique<Impl>();
    impl->type = type;
}


ToneMapper::~ToneMapper() = default;


void
ToneMapper::registerImage(Image<Vector3float>* hdr_image)
{
    //Delete old render buffer if an image has been registered
    if(impl->isRegistered)
    {
        RenderBuffer::destroyDeviceObject(impl->render_buffer);
    }

    impl->render_buffer = RenderBuffer::createDeviceObject(hdr_image->width(), hdr_image->height());
    impl->hdr_image = hdr_image;
    impl->isRegistered = true;
}

void
ToneMapper::toneMap()
{
    if(impl->isRegistered)
    {
        switch(impl->type)
        {
            case REINHARD:
            {
                impl->toneMappingReinhard();
            }
            break;
            case GAMMA:
            {
                impl->toneMappingGamma();
            }
            break;
        }
    }
    else
    {
        std::cerr << "[ToneMapper]  No HDR image has been registered. Call registerImage() first!" << std::endl;
    }
}


RenderBuffer
ToneMapper::getRenderBuffer()
{
    return impl->render_buffer;
}

void
ToneMapper::Impl::toneMappingReinhard()
{
    KernelSizeHelper::KernelSize config = KernelSizeHelper::configure(hdr_image->size());
    detail::reinhard_kernel<<<config.blocks, config.threads>>>(*hdr_image, render_buffer);
    cudaSafeCall(cudaDeviceSynchronize());
}

void
ToneMapper::Impl::toneMappingGamma()
{
    KernelSizeHelper::KernelSize config = KernelSizeHelper::configure(hdr_image->size());
    detail::gamma_kernel<<<config.blocks, config.threads>>>(*hdr_image, render_buffer);
    cudaSafeCall(cudaDeviceSynchronize());
}

void
ToneMapper::saveToFile(const std::string& path)
{
    RenderBuffer host_buffer = RenderBuffer::createHostObject(impl->hdr_image->width(), impl->hdr_image->height());

    impl->render_buffer.copyDevice2HostObject(host_buffer);

    Vector4uint8_t* image_data = host_buffer.data();

    uint8_t* byte_data = (uint8_t*)image_data;

    stbi_flip_vertically_on_write(true);
    stbi_write_bmp(path.c_str(), host_buffer.width(), host_buffer.height(), 4, byte_data);

    RenderBuffer::destroyHostObject(host_buffer);
}

ToneMappingType
ToneMapper::getType()
{
    return impl->type;
}