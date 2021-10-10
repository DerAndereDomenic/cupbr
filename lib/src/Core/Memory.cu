#include <Core/Memory.cuh>
#include <iostream>

namespace cupbr
{
    Memory* Memory::instance = new Memory;

    void
        Memory::printStatisticsImpl()
    {
        printf("Host:   %i/%i (%i)\n", Memory::allocated_host, Memory::deallocated_host, Memory::allocated_host - Memory::deallocated_host);
        printf("Device: %i/%i (%i)\n", Memory::allocated_device, Memory::deallocated_device, Memory::allocated_device - Memory::deallocated_device);
    }
} //namespace cupbr
