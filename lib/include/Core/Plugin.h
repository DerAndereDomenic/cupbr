#ifndef __CUPBR_CORE_PLUGIN_H
#define __CUPBR_CORE_PLUGIN_H

#include <Core/CUPBRAPI.h>
#include <Core/Properties.h>
#include <Core/Memory.h>
#include <Core/CUDA.h>
#include <unordered_map>
#include <memory>

#ifdef CUPBR_WINDOWS
struct HINSTANCE__;
typedef HINSTANCE__* HINSTANCE;
typedef HINSTANCE HMODULE;
#define CUPBR_PLUGIN_FILE_ENDING ".dll"
#endif

#ifdef CUPBR_DEBUG
#define CUPBR_PLUGIN_PATH "bin/Debug"
#else
#define CUPBR_PLUGIN_PATH "bin/Release"
#endif

namespace cupbr
{
    /**
    *   @brief A class to model a plugin
    */
    class Plugin
    {
        public:
        /**
        *   @brief Default constructor
        */
        Plugin() = default;

        /**
        *   @brief Default destructor
        */
        virtual ~Plugin() = default;
    };

    /**
    *   @brief An instance of a plugin
    */
    class PluginInstance
    {
        public:
        /**
        *   @brief Constructor
        *   @param[in] name The name of the plugin (name of the shared library)
        */
        PluginInstance(const std::string& name);

        /**
        *   @brief Destructor
        */
        ~PluginInstance();

        /**
        *   @brief Load an object associated with this plugin instance on the device
        *   Creates a new object that is implemented inside the plugin  (for example the Material class)
        *   @param[in] properties Properties needed for creating the object
        *   @return Device pointer of the plugin instance
        */
        Plugin* createDeviceObject(Properties* properties);

        /**
        *   @brief Load an object associated with this plugin instance on the host
        *   Creates a new object that is implemented inside the plugin  (for example the Material class)
        *   @param[in] properties Properties needed for creating the object
        *   @return Host pointer of the plugin instance
        */
        Plugin* createHostObject(Properties* properties);

        /**
        *   @brief Get the name of the plugin
        *   @return The name
        */
        std::string get_name() const;

        /**
        *   @brief Get the version of the plugin
        *   @return The version
        */
        std::string get_version() const;

        private:
        Plugin* (*_createDeviceObject)(Properties* properties);         /**< Function pointer for loading an instance */
        Plugin* (*_createHostObject)(Properties* properties);           /**< Function pointer for loading an instance */
        char* (*_get_name)();                                           /**< Function pointer for the name */
        char* (*_get_version)();                                        /**< Function pointer for the version */

        #ifdef CUPBR_WINDOWS
        HMODULE* _handle;                                               /**< The library handle for windows */
        #endif
    };

    /**
    *   @brief A static class that handles all plugin
    */
    class PluginManager
    {
        public:
        /**
        *   @brief Load a plugin
        *   @param[in] name The name of the plugin (The same as the shared library file)
        */
        inline static void loadPlugin(const std::string& name) { _instance->loadPluginImpl(name); }

        /**
        *   @brief Get a pointer to the plugin instance
        *   @param[in] name The name of the plugin
        *   @return Plugin instance pointer
        */
        inline static std::shared_ptr<PluginInstance> getPlugin(const std::string& name) { return _instance->getPluginImpl(name); }

        /**
        *   @brief Unload a plugin
        *   @param name The name of the plugin
        */
        inline static void unloadPlugin(const std::string& name) { _instance->unloadPluginImpl(name); }

        /**
        *   @brief Destroy the Manager and all its instances
        */
        inline static void destroy() { _instance->destroyImpl(); }

        //Iterators
        inline static std::unordered_map<std::string, std::shared_ptr<PluginInstance>>::iterator begin() { return _instance->_plugins.begin(); }

        inline static std::unordered_map<std::string, std::shared_ptr<PluginInstance>>::iterator end() { return _instance->_plugins.end(); }

        private:
        static PluginManager* _instance;                                                    /**< The static instance */
        std::unordered_map<std::string, std::shared_ptr<PluginInstance>> _plugins;

        // Implementations
        void loadPluginImpl(const std::string& name);
        std::shared_ptr<PluginInstance> getPluginImpl(const std::string& name);
        void unloadPluginImpl(const std::string& name);
        void destroyImpl();
    };
    /**
    *   @brief Methods that need to be implemented by each plugin
    *   Creates functions for loading, destroying and retreiving names.
    *   Currently we only load device pointer that create consistent vtables for device inheritance
    */
    #define DEFINE_PLUGIN(classType, pluginName, pluginVersion)                                                                     \
    __global__ void fix_vtable(classType** plugin, classType* dummy_plugin)                                                         \
    {                                                                                                                               \
        *plugin = new classType(*dummy_plugin);                                                                                     \
    }                                                                                                                               \
                                                                                                                                    \
    extern "C"                                                                                                                      \
    {                                                                                                                               \
        CUPBR_EXPORT Plugin* createDeviceObject(Properties* properties)                                                             \
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
        CUPBR_EXPORT Plugin* createHostObject(Properties* properties)                                                               \
        {                                                                                                                           \
            return new classType(properties);                                                                                       \
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