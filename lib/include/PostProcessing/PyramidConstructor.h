#ifndef __CUPBR_POSTPROCESSING_PYRAMIDCONSTRUCTOR_H
#define __CUPBR_POSTPROCESSING_PYRAMIDCONSTRUCTOR_H

#include <DataStructure/Image.cuh>

namespace cupbr
{
    namespace PostProcessing
    {
        /**
        *	@brief Threshold the radiance for the bloom effect
        *	@param[in/out] img The image
        *	@param[in] threshold The threshold
        */
        void radiance_threshold(Image<Vector3float>* img, const Vector4float& threshold);

        /**
        *	@brief Construct the image pyramid
        *	@param[in] pyramid (device)
        *	@param[in] host_pyramid (host)
        *	@param[in] pyramid_depth The depth of the pyramid
        */
        void construct_pyramid(Image<Vector3float>* pyramid, Image<Vector3float>* host_pyramid, const uint32_t& pyramid_depth);

        /**
        *	@brief Construct the upsampled pyramid
        *	@param[in] pyramid_down The downsampled pyramid (device)
        *	@param[in] pyramid_up The upsampled pyramid (device)
        *	@param[in] host_pyramid_down The downsampled pyramid (host)
        *	@param[in] host_pyramid_up The upsampled pyramid (host)
        *	@param[in] pyramid_depth The pyramid depth
        *	@param[in] hdr_image The hdr image
        *	@param[out] output The final output image
        */
        void upscale_and_combine(Image<Vector3float>* pyramid_down,
                                 Image<Vector3float>* pyramid_up,
                                 Image<Vector3float>* host_pyramid_down,
                                 Image<Vector3float>* host_pyramid_up,
                                 const uint32_t& pyramid_depth,
                                 const Image<Vector3float>* hdr_image,
                                 Image<Vector3float>* output);
    } //namespace PostProcessing
} //namespace cupbr

#endif