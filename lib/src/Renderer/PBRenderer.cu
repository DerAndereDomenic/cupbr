#include <Renderer/PBRenderer.cuh>

class PBRenderer::Impl
{
    public:
        Impl();

        ~Impl();

        //Data
        RenderingMethod method;
        Image<Vector3float> hdr_image;
        Scene scene;
        uint32_t scene_size;

        bool outputSizeSet;
        bool sceneRegistered;
};

PBRenderer::Impl::Impl()
{
    outputSizeSet = false;
    sceneRegistered = false;
    scene_size = 0;
}

PBRenderer::Impl::~Impl()
{
    Image<Vector3float>::destroyDeviceObject(hdr_image);
}

PBRenderer::PBRenderer(const RenderingMethod& method)
{
    impl = std::make_unique<Impl>();
    impl->method = method;
}

PBRenderer::~PBRenderer() = default;

void
PBRenderer::setOutputSize(const uint32_t& width, const uint32_t& height)
{
    if(impl->outputSizeSet)
    {
        Image<Vector3float>::destroyDeviceObject(impl->hdr_image);
    }

    impl->hdr_image = Image<Vector3float>::createDeviceObject(width, height);
    impl->outputSizeSet = true;
}

void
PBRenderer::registerScene(const Scene scene, const uint32_t& scene_size)
{
    impl->scene = scene;
    impl->scene_size = scene_size;
    impl->sceneRegistered = true;
}

void
PBRenderer::render()
{
    if(!impl->outputSizeSet)
    {
        std::cerr << "[PBRenderer]  No output size set. Use setOutputSize()" << std::endl;
        return;
    }

    if(!impl->sceneRegistered)
    {
        std::cerr << "[PBRenderer]  No scene registered. Use registerScene()" << std::endl;
        return;
    }
}

Image<Vector3float>*
PBRenderer::getOutputImage()
{
    return &impl->hdr_image;
}