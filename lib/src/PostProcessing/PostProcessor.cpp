#include <PostProcessing/PostProcessor.h>
#include <PostProcessing/Convolution.h>
#include <PostProcessing/PyramidConstructor.h>

namespace cupbr
{
    class PostProcessor::Impl
    {
        public:
        Impl();

        ~Impl();

        void buildHierarchyBuffers();
        void destroyHierarchyBuffers();

        Image<Vector3float>* hdr_image;
        Image<Vector3float> output;

        uint32_t pyramid_depth = 6;

        Image<Vector3float>* pyramid_down;
        Image<Vector3float>* host_pyramid_down;

        Image<Vector3float>* pyramid_up;
        Image<Vector3float>* host_pyramid_up;

        bool isRegistered;
    };

    PostProcessor::Impl::Impl()
    {
        isRegistered = false;
    }

    void
    PostProcessor::Impl::buildHierarchyBuffers()
    {
        pyramid_down = Memory::createDeviceArray<Image<Vector3float>>(pyramid_depth);
        host_pyramid_down = Memory::createHostArray<Image<Vector3float>>(pyramid_depth);

        pyramid_up = Memory::createDeviceArray<Image<Vector3float>>(pyramid_depth);
        host_pyramid_up = Memory::createHostArray<Image<Vector3float>>(pyramid_depth);

        uint32_t width = hdr_image->width();
        uint32_t height = hdr_image->height();

        for (uint32_t i = 0; i < pyramid_depth; ++i)
        {
            host_pyramid_down[i] = Image<Vector3float>::createDeviceObject(width, height);
            host_pyramid_up[i] = Image<Vector3float>::createDeviceObject(width, height);
            width /= 2;
            height /= 2;
        }

        hdr_image->copyDevice2DeviceObject(host_pyramid_down[0]);
        Memory::copyHost2DeviceArray<Image<Vector3float>>(pyramid_depth, host_pyramid_down, pyramid_down);
        Memory::copyHost2DeviceArray<Image<Vector3float>>(pyramid_depth, host_pyramid_up, pyramid_up);
    }

    void
    PostProcessor::Impl::destroyHierarchyBuffers()
    {
        Memory::copyDevice2HostArray<Image<Vector3float>>(pyramid_depth, pyramid_down, host_pyramid_down);
        Memory::copyDevice2HostArray<Image<Vector3float>>(pyramid_depth, pyramid_up, host_pyramid_up);

        for (uint32_t i = 0; i < pyramid_depth; ++i)
        {
            Image<Vector3float>::destroyDeviceObject(host_pyramid_down[i]);
            Image<Vector3float>::destroyDeviceObject(host_pyramid_up[i]);
        }
        Memory::destroyDeviceArray<Image<Vector3float>>(pyramid_down);
        Memory::destroyHostArray<Image<Vector3float>>(host_pyramid_down);

        Memory::destroyDeviceArray<Image<Vector3float>>(pyramid_up);
        Memory::destroyHostArray<Image<Vector3float>>(host_pyramid_up);
    }

    PostProcessor::Impl::~Impl()
    {
        if (isRegistered)
        {
            Image<Vector3float>::destroyDeviceObject(output);

            destroyHierarchyBuffers();
        }
        isRegistered = false;
    }

    PostProcessor::PostProcessor()
    {
        impl = std::make_unique<Impl>();
    }

    PostProcessor::~PostProcessor() = default;

    void
    PostProcessor::registerImage(Image<Vector3float>* hdr_image)
    {
        if (impl->isRegistered)
        {
            Image<Vector3float>::destroyDeviceObject(impl->output);

            impl->destroyHierarchyBuffers();
        }

        impl->hdr_image = hdr_image;
        impl->output = Image<Vector3float>::createDeviceObject(hdr_image->width(), hdr_image->height());
        impl->buildHierarchyBuffers();
        impl->isRegistered = true;
    }

    Image<Vector3float>*
    PostProcessor::getPostProcessBuffer()
    {
        return &(impl->output);
    }

    void
    PostProcessor::filter(Image<Vector3float>& kernel)
    {
        PostProcessing::convolve(*(impl->hdr_image), kernel, &(impl->output));
    }

    void
    PostProcessor::bloom(const Vector4float& threshold)
    {
        impl->hdr_image->copyDevice2DeviceObject(impl->host_pyramid_down[0]);
        PostProcessing::radiance_threshold(impl->host_pyramid_down, threshold);
        PostProcessing::construct_pyramid(impl->pyramid_down, impl->host_pyramid_down, impl->pyramid_depth);

        //Copy last downsampled image to upscaling pyramid
        impl->host_pyramid_down[impl->pyramid_depth - 1].copyDevice2DeviceObject(impl->host_pyramid_up[impl->pyramid_depth - 1]);

        PostProcessing::upscale_and_combine(impl->pyramid_down,
                                            impl->pyramid_up,
                                            impl->host_pyramid_down,
                                            impl->host_pyramid_up,
                                            impl->pyramid_depth,
                                            impl->hdr_image,
                                            &(impl->output));
    }

} //namespace cupbr