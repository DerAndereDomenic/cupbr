#ifndef __CUPBR_CORE_PLUGIN_H
#define __CUPBR_CORE_PLUGIN_H

#include <Core/CUPBRAPI.h>
#include <Core/Properties.h>
#include <Core/Memory.h>
#include <Core/CUDA.h>

#ifdef CUPBR_WINDOWS
class HINSTANCE__;
typedef HINSTANCE__* HINSTANCE;
typedef HINSTANCE HMODULE;
#endif

namespace cupbr
{
    /**
    *   @brief A class to model a plugin
    */
    class Plugin
    {
        public:
        Plugin() = default;

        virtual ~Plugin() = default;
    };

    class PluginInstance
    {
        public:
        PluginInstance(const std::string& name);

        ~PluginInstance();

        Plugin* load(const Properties& properties);

        std::string get_name() const;

        std::string get_version() const;

        private:
        Plugin* (*_load)(const Properties& properties);
        char* (*_get_name)();
        char* (*_get_version)();

        Plugin* _instance = nullptr;

        #ifdef CUPBR_WINDOWS
        HMODULE* _handle;
        #endif
    };

    #define DEFINE_PLUGIN(classType, pluginName, pluginVersion)                         \
    template<typename T>                                                                \
    __global__ void fix_vtable(T* object)                                               \
    {                                                                                   \
        T temp(*object);                                                                \
        new (object) T(temp);                                                           \
    }                                                                                   \
                                                                                        \
    extern "C"                                                                          \
    {                                                                                   \
        CUPBR_EXPORT Plugin* load(const Properties& properties)                         \
        {                                                                               \
            classType host_object(properties);                                          \
            classType* plugin = Memory::createDeviceObject<classType>();                \
            Memory::copyHost2DeviceObject<classType>(&host_object, plugin);             \
            fix_vtable << <1, 1 >> > (plugin);                                          \
            cudaSafeCall(cudaDeviceSynchronize());                                      \
            return plugin;                                                              \
        }                                                                               \
                                                                                        \
        CUPBR_EXPORT const char* name()                                                 \
        {                                                                               \
            return pluginName;                                                          \
        }                                                                               \
                                                                                        \
        CUPBR_EXPORT const char* version()                                              \
        {                                                                               \
            return pluginVersion;                                                       \
        }                                                                               \
    }

} //namespace cupbr

#endif