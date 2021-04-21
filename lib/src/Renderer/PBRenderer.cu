#include <Renderer/PBRenderer.cuh>
#include <Renderer/RayTracer.cuh>
#include <Renderer/Whitted.cuh>
#include <Renderer/PathTracer.cuh>

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
        uint32_t frameIndex;

        bool outputSizeSet;
        bool sceneRegistered;
};

PBRenderer::Impl::Impl()
{
    outputSizeSet = false;
    sceneRegistered = false;
    scene_size = 0;
    frameIndex = 0;
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
PBRenderer::render(const Camera& camera)
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

    switch(impl->method)
    {
        case RAYTRACER:
        {
            PBRendering::raytracing(impl->scene,
                                    impl->scene_size,
                                    camera,
                                    &impl->hdr_image);
        }
        break;
        case WHITTED:
        {
            PBRendering::whitted(impl->scene,
                                 impl->scene_size,
                                 camera,
                                 5,
                                 &impl->hdr_image);
        }
        break;
        case PATHTRACER:
        {
            PBRendering::pathtracing(impl->scene,
                                     impl->scene_size,
                                     camera,
                                     impl->frameIndex,
                                     5,
                                     &impl->hdr_image);
            ++impl->frameIndex;
        }
        break;
        case METROPOLIS:
        {
            std::cerr << "[PBRenderer]  METROPOLIS not supported." << std::endl;
        }
        break;
    }
}

Image<Vector3float>*
PBRenderer::getOutputImage()
{
    return &impl->hdr_image;
}