#ifndef RAYTRACER_DATASTRUCTURE_IMPL_IMAGEDETAIL_CUH
#define RAYTRACER_DATASTRUCTURE_IMPL_IMAGEDETAIL_CUH

#include <Core/Memory.cuh>
#include <Core/KernelHelper.cuh>

template<typename T>
Image<T>
Image<T>::createHostObject(const uint32_t width, const uint32_t height)
{
    Image<T> result;
    result._data = Memory::allocator()->createHostArray<T>(width*height);
    result._width = width;
    result._height = height;
    result._size = width*height;

    return result;
}
    
template<typename T>
Image<T>
Image<T>::createDeviceObject(const uint32_t width, const uint32_t height)
{
    Image<T> result;
    result._data = Memory::allocator()->createDeviceArray<T>(width*height);
    result._width = width;
    result._height = height;
    result._size = width*height;
    
    return result;
}

template<typename T>
Image<T>
Image<T>::createHostObject(T* data, const uint32_t width, const uint32_t height)
{
    Image<T> result;
    result._data = Memory::allocator()->createHostArray<T>(width*height);
    Memory::allocator()->copyHost2HostArray<T>(width * height, data, result._data);
    result._width = width;
    result._height = height;
    result._size = width*height;

    return result;
}
    
template<typename T>
Image<T>
Image<T>::createDeviceObject(T* data, const uint32_t width, const uint32_t height)
{
    Image<T> result;
    result._data = Memory::allocator()->createDeviceArray<T>(width*height);
    Memory::allocator()->copyHost2DeviceArray<T>(width * height, data, result._data);
    result._width = width;
    result._height = height;
    result._size = width*height;
    
    return result;
}
    
template<typename T>
void 
Image<T>::destroyHostObject(Image<T>& object)
{
    object._width = 0;
    object._height = 0;
    object._size = 0;
    Memory::allocator()->destroyHostArray<T>(object._data);
}
    
template<typename T>
void 
Image<T>::destroyDeviceObject(Image<T>& object)
{
    object._width = 0;
    object._height = 0;
    object._size = 0;
    Memory::allocator()->destroyDeviceArray<T>(object._data);
}
    
template<typename T>
void
Image<T>::copyHost2HostObject(Image<T>& host_object)
{
    host_object._width = _width;
    host_object._height = _height;
    host_object._size = _size;
    Memory::allocator()->copyHost2HostArray<T>(_size, _data, host_object._data);
}
    
template<typename T>
void
Image<T>::copyHost2DeviceObject(Image<T>& device_object)
{
    device_object._width = _width;
    device_object._height = _height;
    device_object._size = _size;
    Memory::allocator()->copyHost2DeviceArray<T>(_size, _data, device_object._data);
}
    
template<typename T>
void
Image<T>::copyDevice2DeviceObject(Image<T>& device_object)
{
    device_object._width = _width;
    device_object._height = _height;
    device_object._size = _size;
    Memory::allocator()->copyDevice2DeviceArray<T>(_size, _data, device_object._data);
}
    
template<typename T>
void
Image<T>::copyDevice2HostObject(Image<T>& host_object)
{
    host_object._width = _width;
    host_object._height = _height;
    host_object._size = _size;
    Memory::allocator()->copyDevice2HostArray<T>(_size, _data, host_object._data);
}

template<typename T>
__host__ __device__
T&
Image<T>::operator[](unsigned int index)
{
    return _data[index];
}

template<typename T>
__host__ __device__
T&
Image<T>::operator()(Vector2uint32_t& pixel)
{
    unsigned int index = ThreadHelper::pixel2index(pixel, _width);
    return _data[index];
}

template<typename T>
__host__ __device__
T*
Image<T>::data() const
{
    return _data;
}
    
template<typename T>
__host__ __device__
uint32_t
Image<T>::width() const
{
    return _width;
}

template<typename T>
__host__ __device__
uint32_t
Image<T>::height() const
{
    return _height;
}

template<typename T>
__host__ __device__
uint32_t
Image<T>::size() const
{
    return _size;
}

#endif
