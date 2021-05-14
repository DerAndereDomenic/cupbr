#include <Renderer/PBRenderer.cuh>
#include <Renderer/RayTracer.cuh>
#include <Renderer/Whitted.cuh>
#include <Renderer/PathTracer.cuh>
#include <Renderer/GradientDomain.cuh>

namespace detail
{
    __global__ void
    clearBuffer(Image<Vector3float> img)
    {
        const uint32_t tid = ThreadHelper::globalThreadIndex();

        if(tid >= img.size())
        {
            return;
        }

        img[tid] = 0;
    }
}

class PBRenderer::Impl
{
    public:
        Impl();

        ~Impl();

        //Data
        RenderingMethod method;
        Image<Vector3float> hdr_image;
        Image<Vector3float> gradient_x;
        Image<Vector3float> gradient_y;
        Scene scene;
        uint32_t frameIndex;
        uint32_t maxTraceDepth;

        bool outputSizeSet;
        bool sceneRegistered;
};

PBRenderer::Impl::Impl()
{
    outputSizeSet = false;
    sceneRegistered = false;
    frameIndex = 0;
    maxTraceDepth = 5;
}

PBRenderer::Impl::~Impl()
{
    Image<Vector3float>::destroyDeviceObject(hdr_image);
    Image<Vector3float>::destroyDeviceObject(gradient_x);
    Image<Vector3float>::destroyDeviceObject(gradient_y);
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
        Image<Vector3float>::destroyDeviceObject(impl->gradient_x);
        Image<Vector3float>::destroyDeviceObject(impl->gradient_y);
    }

    impl->hdr_image = Image<Vector3float>::createDeviceObject(width, height);
    impl->gradient_x = Image<Vector3float>::createDeviceObject(width, height);
    impl->gradient_y = Image<Vector3float>::createDeviceObject(width, height);
    impl->outputSizeSet = true;
}

void
PBRenderer::registerScene(Scene& scene)
{
    impl->scene = scene;
    impl->sceneRegistered = true;
}

void
PBRenderer::setMethod(const RenderingMethod& method)
{
    impl->method = method;

    impl->frameIndex = 0;

    const KernelSizeHelper::KernelSize config = KernelSizeHelper::configure(impl->hdr_image.size());
    detail::clearBuffer<<<config.blocks, config.threads>>>(impl->hdr_image);
    detail::clearBuffer<<<config.blocks, config.threads>>>(impl->gradient_x);
    detail::clearBuffer<<<config.blocks, config.threads>>>(impl->gradient_y);
    cudaSafeCall(cudaDeviceSynchronize());
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
                                    camera,
                                    &impl->hdr_image);
        }
        break;
        case WHITTED:
        {
            PBRendering::whitted(impl->scene,
                                 camera,
                                 impl->maxTraceDepth-1,
                                 &impl->hdr_image);
        }
        break;
        case PATHTRACER:
        {
            if(camera.moved())
            {
                impl->frameIndex = 0;
            }
            PBRendering::pathtracing(impl->scene,
                                     camera,
                                     impl->frameIndex,
                                     impl->maxTraceDepth,
                                     &impl->hdr_image);
            ++impl->frameIndex;
        }
        break;
        case METROPOLIS:
        {
            std::cerr << "[PBRenderer]  METROPOLIS not supported." << std::endl;
        }
        break;
        case GRADIENTDOMAIN:
        {
            if(camera.moved())
            {
                impl->frameIndex = 0;
            }
            PBRendering::gradientdomain(impl->scene,
                                        camera,
                                        impl->frameIndex,
                                        impl->maxTraceDepth,
                                        &impl->hdr_image,
                                        &impl->gradient_x,
                                        &impl->gradient_y);
            ++impl->frameIndex;
        }
        break;
    }
}

Image<Vector3float>*
PBRenderer::getOutputImage()
{
    return &impl->hdr_image;
}