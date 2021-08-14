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
}

void
PostProcessing::radiance_threshold(Image<Vector3float>* img, const float& threshold)
{
	const KernelSizeHelper::KernelSize config = KernelSizeHelper::configure(img->size());

	detail::thresholding_kernel << <config.blocks, config.threads >> > (*img, threshold);
	cudaDeviceSynchronize();
}