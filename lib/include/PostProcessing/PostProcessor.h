#ifndef __CUPBR_POSTPTROCESSING_POSTPROCESSOR_H
#define __CUPBR_POSTPTROCESSING_POSTPROCESSOR_H

#include <DataStructure/Image.cuh>
#include <memory>

/**
*	@brief A class to model a post processor
*/
class PostProcessor
{
	public:
		/**
		*	@brief Create the post processor class
		*/
		PostProcessor();

		/**
		*	@brief Destructor
		*/
		~PostProcessor();

		/**
		*	@brief Register an image to do the post processing of
		*	@param[in] hdr_image The image
		*/
		void
		registerImage(Image<Vector3float>* hdr_image);

		/**
		*	@brief Get the buffer with the post processed contents
		*	@return The buffer
		*/
		Image<Vector3float>*
		getPostProcessBuffer();
	private:
		class Impl;
		std::unique_ptr<Impl> impl;		/**< The implementation pointer */
};

#endif