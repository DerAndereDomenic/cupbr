#ifndef __CUPBR_CORE_PLATFORM_H
#define __CUPBR_CORE_PLATFORM_H

#ifdef _WIN32
#define CUPBR_WINDOWS
#elif __linux__
#define CUPBR_LINUX
#elif __APPLE__
#define CUPBR_APPLE
#endif

#endif