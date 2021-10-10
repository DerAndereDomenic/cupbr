#ifndef __CUPBR_CORE_KERNELHELPER_CUH
#define __CUPBR_CORE_KERNELHELPER_CUH

#include <Math/Vector.h>

namespace cupbr
{
    /**
    *   @brief A namespace that capsules thread helper functions
    */
    namespace ThreadHelper
    {
        /**
        *   @brief Get the thread index of a kernel thread
        *   @return The thread index
        */
        __device__
            uint32_t
            globalThreadIndex();

        /**
        *   @brief Convertes a 1D index to a 2D pixel
        *   @param[in] index The index
        *   @param[in] width The width of the image
        *   @param[in] height The height of the image
        *   @return The corresponding pixel
        */
        __host__ __device__
            Vector2uint32_t
            index2pixel(const uint32_t& index, const uint32_t& width, const uint32_t& height);

        /**
        *   @brief Converts a 2D pixel to a 1D index
        *   @param[in] pixel The pixel
        *   @param[in] width The width of the image
        *   @return The corresponding index
        */
        __host__ __device__
            uint32_t
            pixel2index(const Vector2uint32_t& pixel, const uint32_t& width);
    } //namespace ThreadHelper

    /**
    *   @brief A namespace for managing kernel sizes
    */
    namespace KernelSizeHelper
    {

        /**
        *   @brief A namespace for Kernel sizes
        */
        struct KernelSize
        {
            const uint32_t THREADS_PER_BLOCK = 128;     /**< The number of threads per block */
            uint32_t blocks;                            /**< The number of blocks */
            uint32_t threads;                           /**< The number of threads */
        };

        /**
        *   @brief Constructs the block/thread size needed for the given size
        *   @param[in] size The overall number of threads needed
        *   @note The returned blocks*threads >= size
        */
        __host__
            KernelSize configure(const uint32_t& size);
    } //namespace KernelSizeHelper

} //namespace cupbr

#include "../../src/Core/KernelHelperDetail.cuh"

#endif
