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

	/**
	*	@brief Construct the image pyramid
	*	@param[in] pyramid (device)
	*	@param[in] host_pyramid (host)
	*	@param[in] pyramid_depth The depth of the pyramid
	*/
	void
	construct_pyramid(Image<Vector3float>* pyramid, Image<Vector3float>* host_pyramid, const uint32_t& pyramid_depth);
}

#endif