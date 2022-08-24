#include <Renderer/PBRenderer.h>
#include <Renderer/RenderMethod.h>

namespace cupbr
{
    namespace detail
    {
        CUPBR_GLOBAL void
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

        RenderMethod* renderer;

        bool outputSizeSet;
        bool sceneRegistered;
    };

    PBRenderer::Impl::Impl()
    {
        outputSizeSet = false;
        sceneRegistered = false;
        frameIndex = 0;

        Properties properties;
        properties.setProperty("max_trace_depth", 5);
        properties.setProperty("use_russian_roulette", true);

        renderer = reinterpret_cast<RenderMethod*>(PluginManager::getPlugin("VolumeRenderer")->createHostObject(&properties));
    }

    PBRenderer::Impl::~Impl()
    {
        Image<Vector3float>::destroyDeviceObject(hdr_image);
        delete renderer;
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
        impl->renderer->render(*(impl->scene),
                              *camera,
                              impl->frameIndex,
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
    PBRenderer::changeRenderMethod(const std::string& name)
    {
        delete impl->renderer;
        impl->renderer = reinterpret_cast<RenderMethod*>(PluginManager::getPlugin(name)->createHostObject(nullptr));
    }

} //namespace cupbr