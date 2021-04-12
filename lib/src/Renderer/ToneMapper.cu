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
};

ToneMapper::Impl::Impl()
{
    isRegistered = false;
}

ToneMapper::Impl::~Impl()
{
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

    }
}

void
ToneMapper::toneMap()
{
    if(impl->isRegistered)
    {

    }
}


RenderBuffer
ToneMapper::getRenderBuffer()
{
    return impl->render_buffer;
}


#endif