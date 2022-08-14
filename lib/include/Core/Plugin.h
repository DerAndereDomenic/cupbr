#ifndef __CUPBR_CORE_PLUGIN_H
#define __CUPBR_CORE_PLUGIN_H

#include <Core/CUPBRAPI.h>
#include <Core/Properties.h>
#include <Core/Memory.h>
#include <Core/CUDA.h>
#include <unordered_map>
#include <memory>

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

        Plugin* load(Properties* properties);

        std::string get_name() const;

        std::string get_version() const;

        private:
        Plugin* (*_load)(Properties* properties);
        char* (*_get_name)();
        char* (*_get_version)();

        #ifdef CUPBR_WINDOWS
        HMODULE* _handle;
        #endif
    };

    class PluginManager
    {
        public:
        inline static void loadPlugin(const std::string& name) { _instance->loadPluginImpl(name); }

        inline static std::shared_ptr<PluginInstance> getPlugin(const std::string& name) { return _instance->getPluginImpl(name); }

        inline static void unloadPlugin(const std::string& name) { _instance->unloadPluginImpl(name); }

        inline static void destroy() { _instance->destroyImpl(); }

        inline static std::unordered_map<std::string, std::shared_ptr<PluginInstance>>::iterator begin() { return _instance->_plugins.begin(); }

        inline static std::unordered_map<std::string, std::shared_ptr<PluginInstance>>::iterator end() { return _instance->_plugins.end(); }

        private:
        static PluginManager* _instance;
        std::unordered_map<std::string, std::shared_ptr<PluginInstance>> _plugins;

        void loadPluginImpl(const std::string& name);
        std::shared_ptr<PluginInstance> getPluginImpl(const std::string& name);
        void unloadPluginImpl(const std::string& name);
        void destroyImpl();
    };

    #define DEFINE_PLUGIN(classType, pluginName, pluginVersion)                                                                     \
    __global__ void fix_vtable(classType** plugin, classType* dummy_plugin)                                                          \
    {                                                                                                                               \
        *plugin = new classType(*dummy_plugin);                                                                                      \
    }                                                                                                                               \
                                                                                                                                    \
    extern "C"                                                                                                                      \
    {                                                                                                                               \
        CUPBR_EXPORT Plugin* load(Properties* properties)                                                                           \
        {                                                                                                                           \
            classType host_object(properties);                                                                                      \
            classType* dummy_plugin;                                                                                                \
            classType** plugin_holder;                                                                                              \
                                                                                                                                    \
            cudaMalloc((void**)&plugin_holder, sizeof(classType**));                                                                \
            cudaMalloc((void**)&dummy_plugin, sizeof(classType));                                                                   \
            cudaMemcpy((void*)dummy_plugin, (void*)&host_object, sizeof(classType), cudaMemcpyHostToDevice);                        \
                                                                                                                                    \
            fix_vtable << <1, 1 >> > (plugin_holder, dummy_plugin);                                                                 \
            cudaSafeCall(cudaDeviceSynchronize());                                                                                  \
                                                                                                                                    \
            classType** host_plugin_dummy = new classType*;                                                                         \
            cudaMemcpy((void*)host_plugin_dummy, (void*)plugin_holder, sizeof(classType**), cudaMemcpyDeviceToHost);                \
            classType* plugin = *host_plugin_dummy;                                                                                 \
                                                                                                                                    \
            delete host_plugin_dummy;                                                                                               \
            cudaFree(plugin_holder);                                                                                                \
            cudaFree(dummy_plugin);                                                                                                 \
            return plugin;                                                                                                          \
        }                                                                                                                           \
                                                                                                                                    \
        CUPBR_EXPORT void destroy(Plugin* plugin)                                                                                   \
        {                                                                                                                           \
            cudaFree(plugin);                                                                                                       \
        }                                                                                                                           \
                                                                                                                                    \
        CUPBR_EXPORT const char* name()                                                                                             \
        {                                                                                                                           \
            return pluginName;                                                                                                      \
        }                                                                                                                           \
                                                                                                                                    \
        CUPBR_EXPORT const char* version()                                                                                          \
        {                                                                                                                           \
            return pluginVersion;                                                                                                   \
        }                                                                                                                           \
    }

} //namespace cupbr

#endif