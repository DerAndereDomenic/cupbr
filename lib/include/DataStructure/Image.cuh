#ifndef __CUPBR_DATASTRUCTURE_IMAGE_CUH
#define __CUPBR_DATASTRUCTURE_IMAGE_CUH

#include <Math/Vector.h>

namespace cupbr
{
    /**
    *   @brief A class to model an image
    *   @tparam The pixel type
    */
    template<typename T>
    class Image
    {
    public:
        /**
        *   @brief Default constructor
        */
        Image() = default;

        /**
        *   @brief Create a host image
        *   @param[in] width The image width
        *   @param[in] height The image height
        *   @return The image
        */
        static
            Image createHostObject(const uint32_t width, const uint32_t height);

        /**
        *   @brief Create a host image with the specified data
        *   @param[in] data The data (host)
        *   @param[in] width The image width
        *   @param[in] height The image height
        */
        static
            Image createHostObject(T* data, const uint32_t width, const uint32_t height);

        /**
        *   @brief Create a device image
        *   @param[in] width The image width
        *   @param[in] height The image height
        *   @return The image
        */
        static
            Image createDeviceObject(const uint32_t width, const uint32_t height);

        /**
        *   @brief Create a device image with the specified data
        *   @param[in] data The data (host)
        *   @param[in] width The image width
        *   @param[in] height The image height
        */
        static
            Image createDeviceObject(T* data, const uint32_t width, const uint32_t height);

        /**
        *   @brief Destroys a host image
        *   @param[in] object The object to be destroyed
        */
        static
            void destroyHostObject(Image<T>& object);

        /**
        *   @brief Destroys a device image
        *   @param[in] object The object to be destroyed
        */
        static
            void destroyDeviceObject(Image<T>& object);

        /**
        *   @brief Copies a host to a host image
        *   @param[in] host_object The target host object
        */
        void
            copyHost2HostObject(Image<T>& host_object);

        /**
        *   @brief Copies a host to a device image
        *   @param[in] device_object The target device object
        */
        void
            copyHost2DeviceObject(Image<T>& device_object);

        /**
        *   @brief Copies a device to a device image
        *   @param[in] device_object The target device object
        */
        void
            copyDevice2DeviceObject(Image<T>& device_object);

        /**
        *   @brief Copies a device to a host image
        *   @param[in] host_object The target host object
        */
        void
            copyDevice2HostObject(Image<T>& host_object);

        /**
        *   @brief Get the value at the specified index
        *   @param[in] index The index
        *   @return The pixel value
        */
        __host__ __device__
            T&
            operator[](unsigned int index);

        /**
        *   @brief Get the value at the specified pixel
        *   @param[in] pixel The pixel
        *   @return The pixel value
        */
        __host__ __device__
            T&
            operator()(Vector2uint32_t& pixel);

        /**
        *   @brief Get a raw pointer to the underlying data
        *   @return The data pointer
        */
        __host__ __device__
            T*
            data() const;

        /**
        *   @brief Get the width of the image
        *   @return The width
        */
        __host__ __device__
            uint32_t
            width() const;

        /**
        *   @brief Get the height of the image
        *   @return The height
        */
        __host__ __device__
            uint32_t
            height() const;

        /**
        *   @brief Get the size of the image
        *   @return The size
        */
        __host__ __device__
            uint32_t
            size() const;

    private:
        uint32_t _width;    /**< The width of the image */
        uint32_t _height;   /**< The height of the image */
        uint32_t _size;     /**< The size of the image */
        T* _data;           /**< The data of the image */
    };

} //namespace cupbr

#include "../../src/DataStructure/ImageDetail.cuh"

#endif
