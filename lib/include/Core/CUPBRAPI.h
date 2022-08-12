#ifndef __CUPBR_CORE_CUPBRAPI_H
#define __CUPBR_CORE_CUPBRAPI_H

#include <Core/Platform.h>

#ifdef CUPBR_WINDOWS
#define CUPBR_EXPORT __declspec(dllexport)
#define CUPBR_IMPORT __declspec(dllimport)
#elif CUPBR_LINUX
#error "Not implemented"
#elif CUPBR_APPLE
#error "Not implemented"
#endif

#endif