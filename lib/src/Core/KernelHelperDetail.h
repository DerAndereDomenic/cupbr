#ifndef __CUPBR_CORE_KERNELHELPERDETAIL_H
#define __CUPBR_CORE_KERNELHELPERDETAIL_H

namespace cupbr
{
    CUPBR_DEVICE
    inline uint32_t
    ThreadHelper::globalThreadIndex()
    {
        return blockIdx.x * blockDim.x + threadIdx.x;
    }

    CUPBR_HOST_DEVICE
    inline Vector2uint32_t
    ThreadHelper::index2pixel(const uint32_t& index, const uint32_t& width, const uint32_t& height)
    {
        uint32_t y = index / width;
        uint32_t x = index - width * y;
        return Vector2uint32_t(x, y);
    }

    CUPBR_HOST_DEVICE
    inline uint32_t
    ThreadHelper::pixel2index(const Vector2uint32_t& pixel, const uint32_t& width)
    {
        return pixel.x + pixel.y * width;
    }


    CUPBR_HOST
    inline KernelSizeHelper::KernelSize
    KernelSizeHelper::configure(const uint32_t& size)
    {
        KernelSizeHelper::KernelSize result;

        result.threads = result.THREADS_PER_BLOCK;


        result.blocks = size / result.THREADS_PER_BLOCK + 1;

        if (size % result.THREADS_PER_BLOCK == 0)
        {
            ++result.blocks;
        }

        return result;
    }

} //namespace cupbr

#endif