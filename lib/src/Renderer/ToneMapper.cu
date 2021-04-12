#include <Renderer/ToneMapper.cuh>

#ifndef __CUPBR_RENDERER_TONEMAPPER_CUH
#define __CUPBR_RENDERER_TONEMAPPER_CUH

#include <DataStructure/Image.cuh>
#include <DataStructure/RenderBuffer.cuh>
#include <memory>

#include <Core/KernelHelper.cuh>
#include <Math/Functions.cuh>

namespace detail
{
    __global__ void
    reinhard_kernel(const Image<Vector3float> hdr_image, RenderBuffer output)
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
}

class ToneMapper::Impl()
{
    public:
        Impl();

        ~Impl();

        /**
        *   @brief The Reinhard tone mapping algorithm
        */
        void
        toneMappingReinhard();

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

ToneMapper::ToneMapper(const ToneMappingType& type = REINHARD)
{
    impl = std::make_unique<Impl>();
    impl->type = type;
}


ToneMapper::~ToneMapper() = default;


void
ToneMapper::registerImage(const Image<Vector3float>* hdr_image)
{
    //Delete old render buffer if an image has been registered
    if(impl->isRegistered)
    {
        RenderBuffer::destroyDeviceObject(impl->render_buffer);
    }

    impl->render_buffer = RenderBuffer::createDeviceObject(hdr_image->width(), hdr_image->height());
    impl->hdr_image = hdr_image;
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
                std::cerr << "[ToneMapper]  GAMMA is not supported at the moment" << std::endl;
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
    KernelSizeHelper::KernelSize config = KernelSizeHelper::configure(hdr_img->size());
    detail::reinhard_kernel<<<config.blocks, config.threads>>>(hdr_image, render_buffer);
    cudaSafeCall(cudaDeviceSynchronize());
}

#endif