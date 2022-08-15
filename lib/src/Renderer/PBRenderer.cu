#include <Renderer/PBRenderer.h>
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
        Image<Vector3float> hdr_image;

        Scene* scene;
        uint32_t frameIndex;
        uint32_t maxTraceDepth;

        VolumeRenderer renderer;

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
    }

    PBRenderer::PBRenderer()
    {
        impl = std::make_unique<Impl>();
    }

    PBRenderer::~PBRenderer() = default;

    void
    PBRenderer::setOutputSize(const uint32_t& width, const uint32_t& height)
    {
        if (impl->outputSizeSet)
        {
            Image<Vector3float>::destroyDeviceObject(impl->hdr_image);
        }

        impl->hdr_image = Image<Vector3float>::createDeviceObject(width, height);
        impl->outputSizeSet = true;
    }

    void
    PBRenderer::registerScene(Scene* scene)
    {
        impl->scene = scene;
        impl->sceneRegistered = true;
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

        if (camera->moved())
        {
            impl->frameIndex = 0;
        }
        impl->renderer.render(*(impl->scene),
                              *camera,
                              impl->frameIndex,
                              impl->maxTraceDepth,
                              impl->useRussianRoulette,
                              &impl->hdr_image);
        ++impl->frameIndex;
    }

    Image<Vector3float>*
    PBRenderer::getOutputImage()
    {
        return &impl->hdr_image;
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

    uint32_t 
    PBRenderer::getMaxTraceDepth()
    {
        return impl->maxTraceDepth;
    }

} //namespace cupbr