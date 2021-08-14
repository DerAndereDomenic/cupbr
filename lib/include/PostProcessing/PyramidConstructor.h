#ifndef __CUPBR_POSTPROCESSING_PYRAMIDCONSTRUCTOR_H
#define __CUPBR_POSTPROCESSING_PYRAMIDCONSTRUCTOR_H

#include <DataStructure/Image.cuh>

namespace PostProcessing
{
	/**
	*	@brief Threshold the radiance for the bloom effect
	*	@param[in/out] img The image
	*	@param[in] threshold The threshold
	*/
	void
	radiance_threshold(Image<Vector3float>* img, const float& threshold);
}

#endif