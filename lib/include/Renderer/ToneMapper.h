#ifndef __CUPBR_RENDERER_TONEMAPPER_H
#define __CUPBR_RENDERER_TONEMAPPER_H

#include <DataStructure/Image.h>
#include <DataStructure/RenderBuffer.h>
#include <memory>

namespace cupbr
{
    /**
    *   @brief A enum to switch between Tone mapping algorithms
    */
    enum class ToneMappingType
    {
        REINHARD,
        GAMMA
    };

    /**
    *   @brief A class that handles tone mapping
    */
    class ToneMapper
    {
        public:
        /**
        *   @brief Create a tone mapper
        *   @param[in] type The tone mapping type
        */
        ToneMapper(const ToneMappingType& type = ToneMappingType::REINHARD);

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
        *   @brief Get the currently selected tone mapping type
        *   @return The tone mapping type
        */
        ToneMappingType getType();

        /**
        *   @brief Change the tone mapping
        *   @param[in] type The new tone mapping type
        */
        void setType(const ToneMappingType& type);

        /**
        *   @brief Set exposure
        *   @param[in] exposure The exposure time of the camera
        */
        void setExposure(const float& exposure);

        /**
        *   @brief Store the current render buffer to file
        *   @param[in] path The output path
        */
        void saveToFile(const std::string& path);

        private:
        class Impl;
        std::unique_ptr<Impl> impl;     /**< The implementation pointer */
    };
} //namespace cupbr

#endif