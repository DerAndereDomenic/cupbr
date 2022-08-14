#include <Core/Memory.h>
#include <iostream>

namespace cupbr
{
    Memory* Memory::instance = new Memory;

    void
    Memory::printStatisticsImpl()
    {
        std::cout << "Host:   " << Memory::allocated_host << "/" << Memory::deallocated_host <<" ("<< Memory::allocated_host - Memory::deallocated_host << ")"  << std::endl;
        std::cout << "Device: " << Memory::allocated_device << "/" << Memory::deallocated_device <<" ("<< Memory::allocated_device - Memory::deallocated_device << ")"  << std::endl;
    }
} //namespace cupbr
