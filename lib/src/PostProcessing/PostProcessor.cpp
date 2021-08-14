#include <PostProcessing/PostProcessor.h>
#include <PostProcessing/Convolution.cuh>

class PostProcessor::Impl
{
	public:
		Impl();

		~Impl();

		Image<Vector3float>* hdr_image;
		Image<Vector3float> output;

		bool isRegistered;
};

PostProcessor::Impl::Impl()
{
	isRegistered = false;
}

PostProcessor::Impl::~Impl()
{
	if(isRegistered)
	{
		Image<Vector3float>::destroyDeviceObject(output);
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
	}

	impl->hdr_image = hdr_image;
	impl->output = Image<Vector3float>::createDeviceObject(hdr_image->width(), hdr_image->height());
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