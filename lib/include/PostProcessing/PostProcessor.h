#ifndef __CUPBR_POSTPTROCESSING_POSTPROCESSOR_H
#define __CUPBR_POSTPTROCESSING_POSTPROCESSOR_H

#include <DataStructure/Image.h>
#include <memory>

namespace cupbr
{
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
        void registerImage(Image<Vector3float>* hdr_image);

        /**
        *	@brief Get the buffer with the post processed contents
        *	@return The buffer
        */
        Image<Vector3float>* getPostProcessBuffer();

        /**
        *	@brief Filter the image using convolution
        *	@param[in] kernel The image kernel
        */
        void filter(Image<Vector3float>& kernel);

        /**
        *	@brief Apply a bloom effect
        *	@param[in] threshold The quadratic threshold curve (threshold, knee - threshold, 2*knee, 0.25/knee)
        */
        void bloom(const Vector4float& threshold);

        private:
        class Impl;
        std::unique_ptr<Impl> impl;		/**< The implementation pointer */
    };

} //namespace cupbr

#endif