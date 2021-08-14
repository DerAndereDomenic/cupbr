#ifndef __CUPBR_POSTPROCESSING_CONVOLUTION_H
#define __CUPBR_POSTPROCESSING_CONVOLUTION_H

#include <DataStructure/Image.cuh>

namespace PostProcessing
{
	/**
	*	@brief Apply convolution on the input
	*	@param[in] input The input image
	*	@param[in] kernel The image kernel
	*	@param[out] The filtered output
	*/
	void
	convolve(Image<Vector3float>& input,
			 Image<Vector3float>& kernel,
			 Image<Vector3float>* output);
}

#endif