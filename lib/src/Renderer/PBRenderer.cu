#include <Renderer/PBRenderer.h>
#include <Renderer/RayTracer.h>
#include <Renderer/Whitted.h>
#include <Renderer/PathTracer.h>
#include <Renderer/GradientDomain.h>
#include <Renderer/VolumeRenderer.h>

namespace cupbr
{
    namespace detail
    {
        __global__ void
        clearBuffer(Image<Vector3float> img)
        {
            const uint32_t tid = ThreadHelper::globalThreadIndex();

            if (tid >= img.size())
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
        Image<Vector3float> gradient_x_forward;
        Image<Vector3float> gradient_y_forward;
        Image<Vector3float> gradient_x_backward;
        Image<Vector3float> gradient_y_backward;
        Image<Vector3float> gradient_x;
        Image<Vector3float> gradient_y;
        Image<Vector3float> base;
        Image<Vector3float> temp;

        Scene* scene;
        uint32_t frameIndex;
        uint32_t maxTraceDepth;

        bool outputSizeSet;
        bool sceneRegistered;
        bool useRussianRoulette;
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
        Image<Vector3float>::destroyDeviceObject(gradient_x_forward);
        Image<Vector3float>::destroyDeviceObject(gradient_y_forward);
        Image<Vector3float>::destroyDeviceObject(gradient_x_backward);
        Image<Vector3float>::destroyDeviceObject(gradient_y_backward);
        Image<Vector3float>::destroyDeviceObject(base);
        Image<Vector3float>::destroyDeviceObject(temp);
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
        if (impl->outputSizeSet)
        {
            Image<Vector3float>::destroyDeviceObject(impl->hdr_image);
            Image<Vector3float>::destroyDeviceObject(impl->gradient_x);
            Image<Vector3float>::destroyDeviceObject(impl->gradient_y);
            Image<Vector3float>::destroyDeviceObject(impl->gradient_x_forward);
            Image<Vector3float>::destroyDeviceObject(impl->gradient_y_forward);
            Image<Vector3float>::destroyDeviceObject(impl->gradient_x_backward);
            Image<Vector3float>::destroyDeviceObject(impl->gradient_y_backward);
            Image<Vector3float>::destroyDeviceObject(impl->base);
            Image<Vector3float>::destroyDeviceObject(impl->temp);
        }

        impl->hdr_image = Image<Vector3float>::createDeviceObject(width, height);
        impl->gradient_x = Image<Vector3float>::createDeviceObject(width, height);
        impl->gradient_y = Image<Vector3float>::createDeviceObject(width, height);
        impl->gradient_x_forward = Image<Vector3float>::createDeviceObject(width, height);
        impl->gradient_y_forward = Image<Vector3float>::createDeviceObject(width, height);
        impl->gradient_x_backward = Image<Vector3float>::createDeviceObject(width, height);
        impl->gradient_y_backward = Image<Vector3float>::createDeviceObject(width, height);
        impl->base = Image<Vector3float>::createDeviceObject(width, height);
        impl->temp = Image<Vector3float>::createDeviceObject(width, height);
        impl->outputSizeSet = true;
    }

    void
    PBRenderer::registerScene(Scene* scene)
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
        detail::clearBuffer << <config.blocks, config.threads >> > (impl->hdr_image);
        detail::clearBuffer << <config.blocks, config.threads >> > (impl->gradient_x);
        detail::clearBuffer << <config.blocks, config.threads >> > (impl->gradient_y);
        detail::clearBuffer << <config.blocks, config.threads >> > (impl->gradient_x_forward);
        detail::clearBuffer << <config.blocks, config.threads >> > (impl->gradient_y_forward);
        detail::clearBuffer << <config.blocks, config.threads >> > (impl->gradient_x_backward);
        detail::clearBuffer << <config.blocks, config.threads >> > (impl->gradient_y_backward);
        detail::clearBuffer << <config.blocks, config.threads >> > (impl->temp);
        cudaSafeCall(cudaDeviceSynchronize());
    }

    void
    PBRenderer::render(Camera* camera)
    {
        if (!impl->outputSizeSet)
        {
            std::cerr << "[PBRenderer]  No output size set. Use setOutputSize()" << std::endl;
            return;
        }

        if (!impl->sceneRegistered)
        {
            std::cerr << "[PBRenderer]  No scene registered. Use registerScene()" << std::endl;
            return;
        }

        switch (impl->method)
        {
            case RenderingMethod::RAYTRACER:
            {
                PBRendering::raytracing(*(impl->scene),
                                        *camera,
                                        &impl->hdr_image);
            }
            break;
            case RenderingMethod::WHITTED:
            {
                std::cerr << "[PBRenderer]  WHITTED currently disabled." << std::endl;
                return;
                /*
                PBRendering::whitted(impl->scene,
                                     camera,
                                     2,
                                     &impl->hdr_image);*/
            }
            break;
            case RenderingMethod::PATHTRACER:
            {
                if (camera->moved())
                {
                    impl->frameIndex = 0;
                }
                PBRendering::pathtracing(*(impl->scene),
                                         *camera,
                                         impl->frameIndex,
                                         impl->maxTraceDepth,
                                         impl->useRussianRoulette,
                                         &impl->hdr_image);
                ++impl->frameIndex;
            }
            break;
            case RenderingMethod::METROPOLIS:
            {
                std::cerr << "[PBRenderer]  METROPOLIS not supported." << std::endl;
            }
            break;
            case RenderingMethod::GRADIENTDOMAIN:
            {
                if (camera->moved())
                {
                    impl->frameIndex = 0;
                }
                PBRendering::gradientdomain(*(impl->scene),
                                            *camera,
                                            impl->frameIndex,
                                            impl->maxTraceDepth,
                                            &impl->base,
                                            &impl->temp,
                                            &impl->gradient_x,
                                            &impl->gradient_y,
                                            &impl->gradient_x_forward,
                                            &impl->gradient_x_backward,
                                            &impl->gradient_y_forward,
                                            &impl->gradient_y_backward,
                                            &impl->hdr_image);
                ++impl->frameIndex;
            }
            break;
            case RenderingMethod::VOLUME:
            {
                if (camera->moved())
                {
                    impl->frameIndex = 0;
                }
                PBRendering::volumetracing(*(impl->scene),
                                           *camera,
                                           impl->frameIndex,
                                           impl->maxTraceDepth,
                                           impl->useRussianRoulette,
                                           &impl->hdr_image);
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

    Image<Vector3float>*
    PBRenderer::getGradientX()
    {
        return &impl->gradient_x;
    }

    Image<Vector3float>*
    PBRenderer::getGradientY()
    {
        return &impl->gradient_y;
    }

    RenderingMethod
    PBRenderer::getMethod()
    {
        return impl->method;
    }

    void
    PBRenderer::reset()
    {
        impl->frameIndex = 0;
    }

    void 
    PBRenderer::setRussianRoulette(const bool& useRussianRoulette)
    {
        impl->useRussianRoulette = useRussianRoulette;
    }

    void 
    PBRenderer::setMaxTraceDepth(const uint32_t& trace_depth)
    {
        impl->maxTraceDepth = trace_depth;
    }

} //namespace cupbr