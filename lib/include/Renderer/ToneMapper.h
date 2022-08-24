#ifndef __CUPBR_RENDERER_TONEMAPPER_H
#define __CUPBR_RENDERER_TONEMAPPER_H

#include <DataStructure/Image.h>
#include <DataStructure/RenderBuffer.h>
#include <Core/Properties.h>
#include <memory>

namespace cupbr
{
    /**
    *   @brief A class that handles tone mapping
    */
    class ToneMapper
    {
        public:
        /**
        *   @brief Create a tone mapper
        */
        ToneMapper();

        /**
        *   @brief Destructor
        */
        ~ToneMapper();

        /**
        *   @brief Register the hdr image that should be tone mapped
        *   @param[in] hdr_image The image to be tonemapped
        */
        void registerImage(Image<Vector3float>* hdr_image);

        /**
        *   @brief Apply the selected tone mapping algorithm to the registered hdr image
        */
        void toneMap();

        /**
        *   @brief Get the resulting render buffer
        *   @note Holds the result from the last toneMap() call
        */
        RenderBuffer getRenderBuffer();

        /**
        *   @brief Store the current render buffer to file
        *   @param[in] path The output path
        */
        void saveToFile(const std::string& path);

        /**
         * @brief Get the Properties of the Tone Mapper
         * @return The properties
         */
        Properties& getProperties();

        /**
         * @brief Reset the tone mapper and reload from properties
         */
        void reset();

        private:
        class Impl;
        std::unique_ptr<Impl> impl;     /**< The implementation pointer */
    };
} //namespace cupbr

#endif