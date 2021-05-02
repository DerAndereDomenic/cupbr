#ifndef __CUPBR_RENDERER_TONEMAPPER_CUH
#define __CUPBR_RENDERER_TONEMAPPER_CUH

#include <DataStructure/Image.cuh>
#include <DataStructure/RenderBuffer.cuh>
#include <memory>

/**
*   @brief A enum to switch between Tone mapping algorithms 
*/
enum ToneMappingType
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
        ToneMapper(const ToneMappingType& type = REINHARD);

        /**
        *   @brief Destructor 
        */
        ~ToneMapper();

        /**
        *   @brief Register the hdr image that should be tone mapped
        *   @param[in] hdr_image The image to be tonemapped 
        */
        void
        registerImage(Image<Vector3float>* hdr_image);

        /**
        *   @brief Apply the selected tone mapping algorithm to the registered hdr image
        */
        void
        toneMap();

        /**
        *   @brief Get the resulting render buffer
        *   @note Holds the result from the last toneMap() call 
        */
        RenderBuffer
        getRenderBuffer();

        /**
        *   @brief Store the current render buffer to file
        *   @param[in] path The output path
        */
        void
        saveToFile(const std::string& path);
    private:
        class Impl;
        std::unique_ptr<Impl> impl;     /**< The implementation pointer */
};

#endif