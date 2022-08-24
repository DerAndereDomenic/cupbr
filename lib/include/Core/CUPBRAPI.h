#ifndef __CUPBR_CORE_CUPBRAPI_H
#define __CUPBR_CORE_CUPBRAPI_H

#include <Core/Platform.h>

#ifdef CUPBR_WINDOWS
#define CUPBR_EXPORT __declspec(dllexport)
#define CUPBR_IMPORT __declspec(dllimport)
#ifndef NDEBUG
#define CUPBR_DEBUG
#endif
#elif defined(CUPBR_LINUX)
#define CUPBR_EXPORT __attribute__((visibility("default")))
#define CUPBR_IMPORT 
#elif defined(CUPBR_APPLE)
#error "Not implemented"
#endif

#endif