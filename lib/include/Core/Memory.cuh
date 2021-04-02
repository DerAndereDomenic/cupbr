#ifndef __CUPBR_CORE_MEMORY_H
#define __CUPBR_CORE_MEMORY_H

#include <Core/CUDA.cuh>

/**
*   @brief A class responsible for memory allocation
*/
class Memory
{
public:
    /**
    *   @brief Get the allocator
    *   @return The allocator
    */
    static Memory* allocator();
    
    /**
    *   @brief Create an object allocated on the CPU (Host)
    *   @tparam The object type
    *   @return A pointer to the object
    */
    template<typename T>
    T* createHostObject();
    
    /**
    *   @brief Create an array allocated ont CPU (Host)
    *   @tparam The object time
    *   @param[in] size The size of the array
    *   @return A pointer to the array
    */
    template<typename T>
    T* createHostArray(const unsigned int size);
    
    /**
    *   @brief Create an object allocated on the GPU (Device)
    *   @tparam The object type
    *   @return A pointer to the object
    */
    template<typename T>
    T* createDeviceObject();
    
    /**
    *   @brief Create an array allocated ont GPU (Device)
    *   @tparam The object time
    *   @param[in] size The size of the array
    *   @return A pointer to the array
    */
    template<typename T>
    T* createDeviceArray(const unsigned int size);
    
    /**
    *   @brief Destroy an object allocated on the CPU (Host)
    *   @tparam The object type
    *   @param[in] object The object to be destroyed
    */
    template<typename T>
    void destroyHostObject(T* object);
    
    /**
    *   @brief Destroy an array allocated on the CPU (Host)
    *   @tparam The object type
    *   @param[in] array The array to be destroyed
    */
    template<typename T>
    void destroyHostArray(T* array);
    
    /**
    *   @brief Destroy an object allocated on the GPU (Device)
    *   @tparam The object type
    *   @param[in] object The object to be destroyed
    */
    template<typename T>
    void destroyDeviceObject(T* object);
    
    /**
    *   @brief Destroy an array allocated on the GPU (Device)
    *   @tparam The object type
    *   @param[in] array The array to be destroyed
    */
    template<typename T>
    void destroyDeviceArray(T* array);
    
    /**
    *   @brief Copies a host object to another host object
    *   @tparam The object type
    *   @param[in] host_object1 The source object
    *   @param[in] host_object2 The target object
    */
    template<typename T>
    void copyHost2HostObject(T* host_object1, T* host_object2);
    
    /**
    *   @brief Copies a host array to another host array
    *   @tparam The object type
    *   @param[in] size The size of the array
    *   @param[in] host_array1 The source array
    *   @param[in] host_array2 The target array
    */
    template<typename T>
    void copyHost2HostArray(const unsigned int size, T* host_array1, T* host_array2);
    
    /**
    *   @brief Copies a host object to a device object
    *   @tparam The object type
    *   @param[in] host_object The source object
    *   @param[in] device_object The target object
    */
    template<typename T>
    void copyHost2DeviceObject(T* host_object, T* device_object);
    
    /**
    *   @brief Copies a host array to a device array
    *   @tparam The object type
    *   @param[in] size The size of the array
    *   @param[in] host_array The source array
    *   @param[in] device_array The target array
    */
    template<typename T>
    void copyHost2DeviceArray(const unsigned int size, T* host_array, T* device_array);
    
    /**
    *   @brief Copies a device object to a host object
    *   @tparam The object type
    *   @param[in] device_object The source object
    *   @param[in] host_object The target object
    */
    template<typename T>
    void copyDevice2HostObject(T* device_object, T* host_object);
    
    /**
    *   @brief Copies a device array to a host array
    *   @tparam The object type
    *   @param[in] size The size of the array
    *   @param[in] device_array The source array
    *   @param[in] host_array The target array
    */
    template<typename T>
    void copyDevice2HostArray(const unsigned int size, T* device_array, T* host_array);
    
    /**
    *   @brief Copies a device object to a device object
    *   @tparam The object type
    *   @param[in] device_object1 The source object
    *   @param[in] device_object2 The target object
    */
    template<typename T>
    void copyDevice2DeviceObject(T* device_object1, T* device_object2);
    
    /**
    *   @brief Copies a device array to a device array
    *   @tparam The object type
    *   @param[in] size The size of the array
    *   @param[in] device_array1 The source array
    *   @param[in] device_array2 The target array
    */
    template<typename T>
    void copyDevice2DeviceArray(const unsigned int size, T* device_array1, T* device_array2);
    
    /**
    *   @brief Prints the statistic of the allocator
    */
    void printStatistics();
private:
    Memory() = default;                     /**< Private default constructor */
    static Memory *instance;                /**< The singleton instance */
    
    unsigned int allocated_host = 0;        /**< The number of allocated host objects */
    unsigned int allocated_device = 0;      /**< The number of allocated device objects */
    unsigned int deallocated_host = 0;      /**< The number of destroyed host objects */
    unsigned int deallocated_device = 0;    /**< The number of destroyed device objects */
};

#include "../../src/Core/MemoryDetail.cuh"

#endif
