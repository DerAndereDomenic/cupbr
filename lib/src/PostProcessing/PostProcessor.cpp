#include <PostProcessing/PostProcessor.h>
#include <PostProcessing/Convolution.cuh>
#include <PostProcessing/PyramidConstructor.h>

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

		Image<Vector3float>* pyramid;
		Image<Vector3float>* host_pyramid;

		float threshold = 1.0f;

		bool isRegistered;
};

PostProcessor::Impl::Impl()
{
	isRegistered = false;
}

void
PostProcessor::Impl::buildHierarchyBuffers()
{
	pyramid = Memory::allocator()->createDeviceArray<Image<Vector3float>>(pyramid_depth);
	host_pyramid = Memory::allocator()->createHostArray<Image<Vector3float>>(pyramid_depth);
	uint32_t width = hdr_image->width();
	uint32_t height = hdr_image->height();

	for(uint32_t i = 0; i < pyramid_depth; ++i)
	{
		host_pyramid[i] = Image<Vector3float>::createDeviceObject(width, height);
		width /= 2;
		height /= 2;
	}

	hdr_image->copyDevice2DeviceObject(host_pyramid[0]);
	Memory::allocator()->copyHost2DeviceArray<Image<Vector3float>>(pyramid_depth, host_pyramid, pyramid);
}

void
PostProcessor::Impl::destroyHierarchyBuffers()
{
	Memory::allocator()->copyDevice2HostArray<Image<Vector3float>>(pyramid_depth, pyramid, host_pyramid);

	for(uint32_t i = 0; i < pyramid_depth; ++i)
	{
		Image<Vector3float>::destroyDeviceObject(host_pyramid[i]);
	}
	Memory::allocator()->destroyDeviceArray<Image<Vector3float>>(pyramid);
	Memory::allocator()->destroyHostArray<Image<Vector3float>>(host_pyramid);
}

PostProcessor::Impl::~Impl()
{
	if(isRegistered)
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
	if(impl->isRegistered)
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
PostProcessor::bloom()
{
	impl->hdr_image->copyDevice2DeviceObject(impl->host_pyramid[0]);
	PostProcessing::radiance_threshold(impl->host_pyramid, impl->threshold);
	PostProcessing::construct_pyramid(impl->pyramid, impl->host_pyramid, impl->pyramid_depth);
}