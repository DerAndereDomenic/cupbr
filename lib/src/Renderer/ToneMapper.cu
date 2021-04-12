#include <Renderer/ToneMapper.cuh>

#ifndef __CUPBR_RENDERER_TONEMAPPER_CUH
#define __CUPBR_RENDERER_TONEMAPPER_CUH

#include <DataStructure/Image.cuh>
#include <DataStructure/RenderBuffer.cuh>
#include <memory>

class ToneMapper::Impl()
{
    public:
        Impl();

        ~Impl();

        ToneMappingType type;

        bool isRegistered;
        RenderBuffer render_buffer;
        Image<Vector3float>* hdr_image;
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

            }
            break;
            case GAMMA:
            {

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


#endif