__device__
inline uint32_t
ThreadHelper::globalThreadIndex()
{
    return blockIdx.x*blockDim.x + threadIdx.x;
}

__host__ __device__
inline Vector2uint32_t
ThreadHelper::index2pixel(const uint32_t& index, const uint32_t& width, const uint32_t& height)
{
    uint32_t y = index/width;
    uint32_t x = index - width*y;
    return Vector2uint32_t(x,y);
}

__host__ __device__
inline uint32_t
ThreadHelper::pixel2index(const Vector2uint32_t& pixel, const uint32_t& width)
{
    return pixel.x + pixel.y*width;
}


__host__
inline KernelSizeHelper::KernelSize
KernelSizeHelper::configure(const uint32_t& size)
{
    KernelSizeHelper::KernelSize result;
    
    result.threads = result.THREADS_PER_BLOCK;
    
    
    result.blocks = size/result.THREADS_PER_BLOCK + 1;
    
    if(size % result.THREADS_PER_BLOCK == 0)
    {
        ++result.blocks;
    }
    
    return result;
}
