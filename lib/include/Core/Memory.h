#ifndef __CUPBR_CORE_MEMORY_H
#define __CUPBR_CORE_MEMORY_H

#include <Core/CUDA.h>

namespace cupbr
{

    /**
    *   @brief A class responsible for memory allocation
    */
    class Memory
    {
        public:

        /**
        *   @brief Create an object allocated on the CPU (Host)
        *   @tparam The object type
        *   @return A pointer to the object
        */
        template<typename T>
        static T* createHostObject() { return instance->createHostObjectImpl<T>(); }

        /**
        *   @brief Create an array allocated ont CPU (Host)
        *   @tparam The object time
        *   @param[in] size The size of the array
        *   @return A pointer to the array
        */
        template<typename T>
        static T* createHostArray(const uint32_t& size) { return instance->createHostArrayImpl<T>(size); }

        /**
        *   @brief Create an object allocated on the GPU (Device)
        *   @tparam The object type
        *   @return A pointer to the object
        */
        template<typename T>
        static T* createDeviceObject() { return instance->createDeviceObjectImpl<T>(); }

        /**
        *   @brief Create an array allocated ont GPU (Device)
        *   @tparam The object time
        *   @param[in] size The size of the array
        *   @return A pointer to the array
        */
        template<typename T>
        static T* createDeviceArray(const uint32_t& size) { return instance->createDeviceArrayImpl<T>(size); }

        /**
        *   @brief Destroy an object allocated on the CPU (Host)
        *   @tparam The object type
        *   @param[in] object The object to be destroyed
        */
        template<typename T>
        static void destroyHostObject(T* object) { instance->destroyHostObjectImpl<T>(object); }

        /**
        *   @brief Destroy an array allocated on the CPU (Host)
        *   @tparam The object type
        *   @param[in] array The array to be destroyed
        */
        template<typename T>
        static void destroyHostArray(T* array) { instance->destroyHostArrayImpl<T>(array); }

        /**
        *   @brief Destroy an object allocated on the GPU (Device)
        *   @tparam The object type
        *   @param[in] object The object to be destroyed
        */
        template<typename T>
        static void destroyDeviceObject(T* object) { instance->destroyDeviceObjectImpl<T>(object); }

        /**
        *   @brief Destroy an array allocated on the GPU (Device)
        *   @tparam The object type
        *   @param[in] array The array to be destroyed
        */
        template<typename T>
        static void destroyDeviceArray(T* array) { instance->destroyDeviceArrayImpl<T>(array); }

        /**
        *   @brief Copies a host object to another host object
        *   @tparam The object type
        *   @param[in] host_object1 The source object
        *   @param[in] host_object2 The target object
        */
        template<typename T>
        static void copyHost2HostObject(T* host_object1, T* host_object2) { instance->copyHost2HostObjectImpl<T>(host_object1, host_object2); }

        /**
        *   @brief Copies a host array to another host array
        *   @tparam The object type
        *   @param[in] size The size of the array
        *   @param[in] host_array1 The source array
        *   @param[in] host_array2 The target array
        */
        template<typename T>
        static void copyHost2HostArray(const uint32_t& size, T* host_array1, T* host_array2) { instance->copyHost2HostArrayImpl<T>(size, host_array1, host_array2); }

        /**
        *   @brief Copies a host object to a device object
        *   @tparam The object type
        *   @param[in] host_object The source object
        *   @param[in] device_object The target object
        */
        template<typename T>
        static void copyHost2DeviceObject(T* host_object, T* device_object) { instance->copyHost2DeviceObjectImpl<T>(host_object, device_object); }

        /**
        *   @brief Copies a host array to a device array
        *   @tparam The object type
        *   @param[in] size The size of the array
        *   @param[in] host_array The source array
        *   @param[in] device_array The target array
        */
        template<typename T>
        static void copyHost2DeviceArray(const uint32_t& size, T* host_array, T* device_array) { instance->copyHost2DeviceArrayImpl<T>(size, host_array, device_array); }

        /**
        *   @brief Copies a device object to a host object
        *   @tparam The object type
        *   @param[in] device_object The source object
        *   @param[in] host_object The target object
        */
        template<typename T>
        static void copyDevice2HostObject(T* device_object, T* host_object) { instance->copyDevice2HostObjectImpl<T>(device_object, host_object); }

        /**
        *   @brief Copies a device array to a host array
        *   @tparam The object type
        *   @param[in] size The size of the array
        *   @param[in] device_array The source array
        *   @param[in] host_array The target array
        */
        template<typename T>
        static void copyDevice2HostArray(const uint32_t& size, T* device_array, T* host_array) { instance->copyDevice2HostArrayImpl<T>(size, device_array, host_array); }

        /**
        *   @brief Copies a device object to a device object
        *   @tparam The object type
        *   @param[in] device_object1 The source object
        *   @param[in] device_object2 The target object
        */
        template<typename T>
        static void copyDevice2DeviceObject(T* device_object1, T* device_object2) { instance->copyDevice2DeviceObjectImpl<T>(device_object1, device_object2); }

        /**
        *   @brief Copies a device array to a device array
        *   @tparam The object type
        *   @param[in] size The size of the array
        *   @param[in] device_array1 The source array
        *   @param[in] device_array2 The target array
        */
        template<typename T>
        static void copyDevice2DeviceArray(const uint32_t& size, T* device_array1, T* device_array2) { instance->copyDevice2DeviceArrayImpl<T>(size, device_array1, device_array2); }

        /**
        *   @brief Prints the statistic of the allocator
        */
        static void printStatistics() { instance->printStatisticsImpl(); }
        private:

        template<typename T>
        T* createHostObjectImpl();

        template<typename T>
        T* createHostArrayImpl(const uint32_t& size);

        template<typename T>
        T* createDeviceObjectImpl();

        template<typename T>
        T* createDeviceArrayImpl(const uint32_t& size);

        template<typename T>
        void destroyHostObjectImpl(T* object);

        template<typename T>
        void destroyHostArrayImpl(T* array);

        template<typename T>
        void destroyDeviceObjectImpl(T* object);

        template<typename T>
        void destroyDeviceArrayImpl(T* array);

        template<typename T>
        void copyHost2HostObjectImpl(T* host_object1, T* host_object2);

        template<typename T>
        void copyHost2HostArrayImpl(const uint32_t& size, T* host_array1, T* host_array2);

        template<typename T>
        void copyHost2DeviceObjectImpl(T* host_object, T* device_object);

        template<typename T>
        void copyHost2DeviceArrayImpl(const uint32_t& size, T* host_array, T* device_array);

        template<typename T>
        void copyDevice2HostObjectImpl(T* device_object, T* host_object);

        template<typename T>
        void copyDevice2HostArrayImpl(const uint32_t& size, T* device_array, T* host_array);

        template<typename T>
        void copyDevice2DeviceObjectImpl(T* device_object1, T* device_object2);

        template<typename T>
        void copyDevice2DeviceArrayImpl(const uint32_t& size, T* device_array1, T* device_array2);

        void printStatisticsImpl();

        Memory() = default;                     /**< Private default constructor */
        static Memory* instance;                /**< The singleton instance */

        uint32_t allocated_host = 0;            /**< The number of allocated host objects */
        uint32_t allocated_device = 0;          /**< The number of allocated device objects */
        uint32_t deallocated_host = 0;          /**< The number of destroyed host objects */
        uint32_t deallocated_device = 0;        /**< The number of destroyed device objects */
    };

} //namespace cupbr

#include "../../src/Core/MemoryDetail.h"

#endif
