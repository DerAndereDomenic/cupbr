#ifndef __CUPBR_CORE_CUDA_H
#define __CUPBR_CORE_CUDA_H

#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <Core/CUPBRAPI.h>

inline
void check(cudaError_t error, char const* const func, const char* const file, int const line)
{
    if (error != cudaSuccess)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
                static_cast<unsigned int>(error), cudaGetErrorString(error), func);
    }
}

#ifdef CUPBR_DEBUG
#define cudaSafeCall(val) check((val), #val, __FILE__, __LINE__)
#else
#define cudaSafeCall(val) val
#endif

#define setDevice(dev) cudaSafeCall(cudaSetDevice(dev))
#define setDefaultDevice() setDevice(0)

#define synchronizeDefaultStream() cudaSafeCall(cudaDeviceSynchronize())

#define CUPBR_HOST __host__
#define CUPBR_DEVICE __device__
#define CUPBR_HOST_DEVICE __host__ __device__
#define CUPBR_GLOBAL __global__

#endif
