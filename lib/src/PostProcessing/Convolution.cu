#include <PostProcessing/Convolution.cuh>
#include <Core/KernelHelper.cuh>
#include <Core/CUDA.cuh>

namespace cupbr
{
	namespace detail
	{
		__global__
			void convolution_kernel(Image<Vector3float> input,
				Image<Vector3float> kernel,
				Image<Vector3float> output)
		{
			const uint32_t tid = ThreadHelper::globalThreadIndex();

			if (tid >= input.size())
			{
				return;
			}

			Vector2uint32_t pixel = ThreadHelper::index2pixel(tid, input.width(), input.height());

			const int32_t kernel_width_half = kernel.width() / 2;
			const int32_t kernel_height_half = kernel.height() / 2;

			if (pixel.x < kernel_width_half || pixel.x >= input.width() - kernel_width_half || pixel.y < kernel_height_half || pixel.y >= input.height() - kernel_height_half)
			{
				return;
			}

			Vector3float result = 0;
            Vector2uint32_t kernel_lookup;
            Vector2uint32_t image_lookup;

			for (int32_t u = -kernel_width_half; u <= kernel_width_half; ++u)
			{
				for (int32_t v = -kernel_height_half; v <= kernel_height_half; ++v)
				{
                    kernel_lookup = Vector2uint32_t(kernel_width_half + u, kernel_height_half + v);
                    image_lookup = Vector2uint32_t(pixel.x + u, pixel.y + v);
					Vector3float kernel_value = kernel(kernel_lookup);
					Vector3float image_value = input(image_lookup);

					result += kernel_value * image_value;
				}
			}

			output[tid] = result;
		}
	} //namespace detail

	void
		PostProcessing::convolve(Image<Vector3float>& input,
			Image<Vector3float>& kernel,
			Image<Vector3float>* output)
	{
		const KernelSizeHelper::KernelSize config = KernelSizeHelper::configure(input.size());

		detail::convolution_kernel << <config.blocks, config.threads >> > (input, kernel, *output);
		cudaDeviceSynchronize();
	}

} //namespace cupbr
