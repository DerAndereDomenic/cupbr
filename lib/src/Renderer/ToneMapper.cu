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
};

ToneMapper::Impl::Impl()
{

}

ToneMapper::Impl::~Impl()
{

}

ToneMapper::ToneMapper(const ToneMappingType& type = REINHARD)
{

}


ToneMapper::~ToneMapper() = default;


void
ToneMapper::registerImage(const Image<Vector3float>* hdr_image)
{

}

void
ToneMapper::toneMap()
{

}


RenderBuffer
ToneMapper::getRenderBuffer()
{

}


#endif