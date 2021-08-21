#include <PostProcessing/PyramidConstructor.h>
#include <Core/KernelHelper.cuh>
#include <Math/Functions.cuh>

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

		/*Vector3float color = img[tid];

		float red = color.x > threshold ? color.x : 0.0f;
		float green = color.y > threshold ? color.y : 0.0f;
		float blue = color.z > threshold ? color.z : 0.0f;

		img[tid] = Vector3float(red, green, blue);*/

		Vector3float color = img[tid];
		float knee = 1.0f;

		Vector3float curve(threshold - knee, knee * 2, 0.25 / knee);

		float brightness = color.x > color.y ? color.x : color.y;
		brightness = color.z > brightness ? color.z : brightness;

		float rq = Math::clamp(brightness - curve.x, 0.0f, curve.y);
		rq = curve.z * rq * rq;

		color *= fmaxf(rq, brightness - threshold) / fmaxf(brightness, 1e-5f);

		img[tid] = color;
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

	__global__ void
	upscale_and_combine_kernel(Image<Vector3float>* pyramid_down, Image<Vector3float>* pyramid_up, const uint32_t depth)
	{
		const uint32_t tid = ThreadHelper::globalThreadIndex();

		Image<Vector3float> new_image = pyramid_up[depth];

		if(tid >= new_image.size())
		{
			return;
		}

		Image<Vector3float> last_image = pyramid_down[depth + 1];
		Image<Vector3float> current_image = pyramid_down[depth];

		Vector3float result = 0;

		Vector2int32_t pixel = ThreadHelper::index2pixel(tid, current_image.width(), current_image.height());
		Vector2int32_t downsampled = pixel / 2;

		float kernel[3][3] =
		{
			1.0f, 2.0f, 1.0f,
			2.0f, 4.0f, 2.0f,
			1.0f, 2.0f, 1.0f
		};

		for(int32_t u = -1; u < 2; ++u)
		{
			for(int32_t v = -1; v < 2; ++v)
			{
				Vector2int32_t sample = downsampled + Vector2int32_t(u, v);

				if (sample.x < 0)sample.x = 0;
				if (sample.x >= last_image.width())sample.x = last_image.width() - 1;
				if (sample.y < 0)sample.y = 0;
				if (sample.y >= last_image.height())sample.y = last_image.height() - 1;

				result += last_image(static_cast<Vector2uint32_t>(sample)) * kernel[u + 1][v + 1];
			}
		}

		new_image[tid] = result / 16.0f + current_image[tid];
	}

	__global__ void
	combine_kernel(Image<Vector3float> hdr_image, Image<Vector3float> last_image, Image<Vector3float> output)
	{
		const uint32_t tid = ThreadHelper::globalThreadIndex();

		if(tid >= output.size())
		{
			return;
		}

		output[tid] = hdr_image[tid] + last_image[tid];
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

void
PostProcessing::upscale_and_combine(Image<Vector3float>* pyramid_down,
									Image<Vector3float>* pyramid_up,
									Image<Vector3float>* host_pyramid_down,
									Image<Vector3float>* host_pyramid_up,
									const uint32_t& pyramid_depth,
									const Image<Vector3float>* hdr_image,
									Image<Vector3float>* output)
{
	for(int32_t i = pyramid_depth - 2; i >= 0; --i)
	{
		const KernelSizeHelper::KernelSize config = KernelSizeHelper::configure(host_pyramid_up[i].size());

		detail::upscale_and_combine_kernel << <config.blocks, config.threads >> > (pyramid_down, pyramid_up, i);
		cudaDeviceSynchronize();
	}

	const KernelSizeHelper::KernelSize config = KernelSizeHelper::configure(output->size());
	detail::combine_kernel << <config.blocks, config.threads >> > (*hdr_image, host_pyramid_up[0], *output);
	cudaDeviceSynchronize();
}