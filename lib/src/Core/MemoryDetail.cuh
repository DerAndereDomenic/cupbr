#ifndef __CUPBR_CORE_IMPL_MEMORY_DETAIL_CUH
#define __CUPBR_CORE_IMPL_MEMORY_DETAIL_CUH

namespace cupbr
{
    template<typename T>
    T*
        Memory::createHostObjectImpl()
    {
        ++Memory::allocated_host;
        return new T();
    }

    template<typename T>
    T*
        Memory::createHostArrayImpl(const uint32_t& size)
    {
        ++Memory::allocated_host;
        return new T[size];
    }

    template<typename T>
    T*
        Memory::createDeviceObjectImpl()
    {
        T* device_object;
        cudaSafeCall(cudaMalloc((void**)&device_object, sizeof(T)));
        ++Memory::allocated_device;
        return device_object;
    }

    template<typename T>
    T*
        Memory::createDeviceArrayImpl(const uint32_t& size)
    {
        T* device_object;
        cudaSafeCall(cudaMalloc((void**)&device_object, size * sizeof(T)));
        ++Memory::allocated_device;
        return device_object;
    }

    template<typename T>
    void
        Memory::destroyHostObjectImpl(T* object)
    {
        ++Memory::deallocated_host;
        delete object;
    }

    template<typename T>
    void
        Memory::destroyHostArrayImpl(T* array)
    {
        ++Memory::deallocated_host;
        delete[] array;
    }

    template<typename T>
    void
        Memory::destroyDeviceObjectImpl(T* object)
    {
        ++Memory::deallocated_device;
        cudaSafeCall(cudaFree(object));
    }

    template<typename T>
    void
        Memory::destroyDeviceArrayImpl(T* array)
    {
        ++Memory::deallocated_device;
        cudaSafeCall(cudaFree(array));
    }

    template<typename T>
    void
        Memory::copyHost2HostObjectImpl(T* host_object1, T* host_object2)
    {
        cudaSafeCall(cudaMemcpy(host_object2, host_object1, sizeof(T), cudaMemcpyHostToHost));
    }

    template<typename T>
    void
        Memory::copyHost2HostArrayImpl(const uint32_t& size, T* host_array1, T* host_array2)
    {
        cudaSafeCall(cudaMemcpy(host_array2, host_array1, size * sizeof(T), cudaMemcpyHostToHost));
    }

    template<typename T>
    void
        Memory::copyHost2DeviceObjectImpl(T* host_object, T* device_object)
    {
        cudaSafeCall(cudaMemcpy(device_object, host_object, sizeof(T), cudaMemcpyHostToDevice));
    }

    template<typename T>
    void
        Memory::copyHost2DeviceArrayImpl(const uint32_t& size, T* host_array, T* device_array)
    {
        cudaSafeCall(cudaMemcpy(device_array, host_array, size * sizeof(T), cudaMemcpyHostToDevice));
    }

    template<typename T>
    void
        Memory::copyDevice2HostObjectImpl(T* device_object, T* host_object)
    {
        cudaSafeCall(cudaMemcpy(host_object, device_object, sizeof(T), cudaMemcpyDeviceToHost));
    }

    template<typename T>
    void
        Memory::copyDevice2HostArrayImpl(const uint32_t& size, T* device_array, T* host_array)
    {
        cudaSafeCall(cudaMemcpy(host_array, device_array, size * sizeof(T), cudaMemcpyDeviceToHost));
    }

    template<typename T>
    void
        Memory::copyDevice2DeviceObjectImpl(T* device_object1, T* device_object2)
    {
        cudaSafeCall(cudaMemcpy(device_object2, device_object1, sizeof(T), cudaMemcpyDeviceToDevice));
    }

    template<typename T>
    void
        Memory::copyDevice2DeviceArrayImpl(const uint32_t& size, T* device_array1, T* device_array2)
    {
        cudaSafeCall(cudaMemcpy(device_array2, device_array1, size * sizeof(T), cudaMemcpyDeviceToDevice));
    }
} //namespace cupbr

#endif
