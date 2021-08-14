#include <PostProcessing/PyramidConstructor.h>
#include <Core/KernelHelper.cuh>

namespace detail
{
	__global__ void
	thresholding_kernel(Image<Vector3float> img, const float threshold)
	{
		const uint32_t tid = ThreadHelper::globalThreadIndex();

		if(tid >= img.size())
		{
			return;
		}

		Vector3float color = img[tid];

		float red = color.x > threshold ? color.x : 0.0f;
		float green = color.y > threshold ? color.y : 0.0f;
		float blue = color.z > threshold ? color.z : 0.0f;

		img[tid] = Vector3float(red, green, blue);
	}

	__global__ void
	downsample_kernel(Image<Vector3float>* pyramid, const uint32_t depth)
	{
		const uint32_t tid = ThreadHelper::globalThreadIndex();

		Image<Vector3float> current_image = pyramid[depth];

		if(tid >= current_image.size())
		{
			return;
		}

		Image<Vector3float> last_image = pyramid[depth - 1];
		Vector2uint32_t pixel = ThreadHelper::index2pixel(tid, current_image.width(), current_image.height());
		Vector2uint32_t upsampled = 2u * pixel;

		Vector3float result = 0;
		Vector2uint32_t sample;

		#define READ_VALUE(u, v)\
			sample = upsampled + Vector2uint32_t(u,v);\
			if(sample.x >= last_image.width())sample.x  = last_image.width()-1;\
			if(sample.y >= last_image.height())sample.y = last_image.height()-1;\
			result += last_image(sample);

		READ_VALUE(0, 0);
		READ_VALUE(0, 1);
		READ_VALUE(1, 0);
		READ_VALUE(1, 1);

		#undef READ_VALUE
			
		current_image[tid] = result / 4.0f;
	}
}

void
PostProcessing::radiance_threshold(Image<Vector3float>* img, const float& threshold)
{
	const KernelSizeHelper::KernelSize config = KernelSizeHelper::configure(img->size());

	detail::thresholding_kernel << <config.blocks, config.threads >> > (*img, threshold);
	cudaDeviceSynchronize();
}

void
PostProcessing::construct_pyramid(Image<Vector3float>* pyramid, Image<Vector3float>* host_pyramid, const uint32_t& pyramid_depth)
{
	for(uint32_t i = 1; i < pyramid_depth; ++i)
	{
		const KernelSizeHelper::KernelSize config = KernelSizeHelper::configure(host_pyramid[i].size());

		detail::downsample_kernel << <config.blocks, config.threads >> > (pyramid, i);
		cudaDeviceSynchronize();
	}
}