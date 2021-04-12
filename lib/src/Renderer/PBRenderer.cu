#include <Renderer/PBRenderer.cuh>

class PBRenderer::Impl
{
    public:
        Impl();

        ~Impl();
};

PBRenderer::Impl::Impl()
{

}

PBRenderer::Impl::~Impl()
{

}

PBRenderer::PBRenderer(const RenderingMethod& method)
{

}

PBRenderer::~PBRenderer() = default;

void
PBRenderer::setOutputSize(const uint32_t& width, const uint32_t& height)
{

}

void
PBRenderer::registerScene(const Scene scene, const uint32_t& scene_size)
{

}

void
PBRenderer::render()
{

}

Vector3float
PBRenderer::getOutputImage()
{

}