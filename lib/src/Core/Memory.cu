#include <Core/Memory.cuh>
#include <iostream>

namespace cupbr
{
    Memory* Memory::instance = nullptr;

    Memory*
        Memory::allocator()
    {
        if (instance == nullptr)
        {
            instance = new Memory();
        }
        return instance;
    }

    void
        Memory::printStatistics()
    {
        printf("Host:   %i/%i (%i)\n", Memory::allocated_host, Memory::deallocated_host, Memory::allocated_host - Memory::deallocated_host);
        printf("Device: %i/%i (%i)\n", Memory::allocated_device, Memory::deallocated_device, Memory::allocated_device - Memory::deallocated_device);
    }
} //namespace cupbr
