#ifndef __CUPBR_CORE_IMPL_MEMORY_DETAIL_CUH
#define __CUPBR_CORE_IMPL_MEMORY_DETAIL_CUH

namespace cupbr
{
    template<typename T>
    T*
        Memory::createHostObject()
    {
        ++Memory::allocated_host;
        return new T();
    }

    template<typename T>
    T*
        Memory::createHostArray(const uint32_t& size)
    {
        ++Memory::allocated_host;
        return new T[size];
    }

    template<typename T>
    T*
        Memory::createDeviceObject()
    {
        T* device_object;
        cudaSafeCall(cudaMalloc((void**)&device_object, sizeof(T)));
        ++Memory::allocated_device;
        return device_object;
    }

    template<typename T>
    T*
        Memory::createDeviceArray(const uint32_t& size)
    {
        T* device_object;
        cudaSafeCall(cudaMalloc((void**)&device_object, size * sizeof(T)));
        ++Memory::allocated_device;
        return device_object;
    }

    template<typename T>
    void
        Memory::destroyHostObject(T* object)
    {
        ++Memory::deallocated_host;
        delete object;
    }

    template<typename T>
    void
        Memory::destroyHostArray(T* array)
    {
        ++Memory::deallocated_host;
        delete[] array;
    }

    template<typename T>
    void
        Memory::destroyDeviceObject(T* object)
    {
        ++Memory::deallocated_device;
        cudaSafeCall(cudaFree(object));
    }

    template<typename T>
    void
        Memory::destroyDeviceArray(T* array)
    {
        ++Memory::deallocated_device;
        cudaSafeCall(cudaFree(array));
    }

    template<typename T>
    void
        Memory::copyHost2HostObject(T* host_object1, T* host_object2)
    {
        cudaSafeCall(cudaMemcpy(host_object2, host_object1, sizeof(T), cudaMemcpyHostToHost));
    }

    template<typename T>
    void
        Memory::copyHost2HostArray(const uint32_t& size, T* host_array1, T* host_array2)
    {
        cudaSafeCall(cudaMemcpy(host_array2, host_array1, size * sizeof(T), cudaMemcpyHostToHost));
    }

    template<typename T>
    void
        Memory::copyHost2DeviceObject(T* host_object, T* device_object)
    {
        cudaSafeCall(cudaMemcpy(device_object, host_object, sizeof(T), cudaMemcpyHostToDevice));
    }

    template<typename T>
    void
        Memory::copyHost2DeviceArray(const uint32_t& size, T* host_array, T* device_array)
    {
        cudaSafeCall(cudaMemcpy(device_array, host_array, size * sizeof(T), cudaMemcpyHostToDevice));
    }

    template<typename T>
    void
        Memory::copyDevice2HostObject(T* device_object, T* host_object)
    {
        cudaSafeCall(cudaMemcpy(host_object, device_object, sizeof(T), cudaMemcpyDeviceToHost));
    }

    template<typename T>
    void
        Memory::copyDevice2HostArray(const uint32_t& size, T* device_array, T* host_array)
    {
        cudaSafeCall(cudaMemcpy(host_array, device_array, size * sizeof(T), cudaMemcpyDeviceToHost));
    }

    template<typename T>
    void
        Memory::copyDevice2DeviceObject(T* device_object1, T* device_object2)
    {
        cudaSafeCall(cudaMemcpy(device_object2, device_object1, sizeof(T), cudaMemcpyDeviceToDevice));
    }

    template<typename T>
    void
        Memory::copyDevice2DeviceArray(const uint32_t& size, T* device_array1, T* device_array2)
    {
        cudaSafeCall(cudaMemcpy(device_array2, device_array1, size * sizeof(T), cudaMemcpyDeviceToDevice));
    }
} //namespace cupbr

#endif
