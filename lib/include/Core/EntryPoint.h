#ifndef __CUPBR_CORE_ENTRYPOINT_H
#define __CUPBR_CORE_ENTRYPOINT_H

#include <CUPBR.h>

extern int run(int argc, char* argv[]);

int main(int argc, char* argv[])
{
    int exit =  run(argc, argv);
    cupbr::Memory::printStatistics();
    return exit;
}

#endif