#include <Core/Plugin.h>

#ifdef CUPBR_WINDOWS
#include <Windows.h>
#elif defined(CUPBR_LINUX)
#include <dlfcn.h>
#endif

namespace cupbr
{
    PluginManager* PluginManager::_instance = new PluginManager;
    
    PluginInstance::PluginInstance(const std::string& name)
    {
        #ifdef CUPBR_WINDOWS

        _handle = (HMODULE*)malloc(sizeof(HMODULE));
        *_handle = LoadLibrary((name + ".dll").c_str());
        if(*_handle == NULL)
        {
            std::cerr << "Error, could not find plugin " << name << std::endl;
            return;
        }

        _createDeviceObject = (Plugin * (*)(Properties* properties))GetProcAddress(*_handle, "createDeviceObject");
        _createHostObject = (Plugin * (*)(Properties* properties))GetProcAddress(*_handle, "createHostObject");
        _get_name = (char* (*)())GetProcAddress(*_handle, "name");
        _get_version = (char* (*)())GetProcAddress(*_handle, "version");
        _get_super_name = (char* (*)())GetProcAddress(*_handle, "superName");

        std::cout << "Loaded plugin: " << get_name() << "; Version: " << get_version() << std::endl;

        #elif defined(CUPBR_LINUX)
        _handle = dlopen((std::string(CUPBR_PLUGIN_PATH) + "/" + name + ".so").c_str(), RTLD_NOW);
        if(!_handle)
        {
            std::cerr << "Error, could not find plugin " << name << std::endl;
            return;
        }

        _createDeviceObject = (Plugin * (*)(Properties* properties))dlsym(_handle, "createDeviceObject");
        _createHostObject = (Plugin * (*)(Properties* properties))dlsym(_handle, "createHostObject");
        _get_name = (char* (*)())dlsym(_handle, "name");
        _get_version = (char* (*)())dlsym(_handle, "version");
        _get_super_name = (char* (*)())dlsym(_handle, "superName");

        std::cout << "Loaded plugin: " << get_name() << "; Version: " << get_version() << std::endl;

        #endif
    }

    PluginInstance::~PluginInstance()
    {
        #ifdef CUPBR_WINDOWS
        FreeLibrary(*_handle);
        free(_handle);
        #elif defined(CUPBR_LINUX)
        dlclose(_handle);
        #endif
    }

    Plugin* 
    PluginInstance::createDeviceObject(Properties* properties)
    {
        return _createDeviceObject(properties);
    }

    Plugin* 
    PluginInstance::createHostObject(Properties* properties)
    {
        return _createHostObject(properties);
    }

    std::string 
    PluginInstance::get_name() const
    {
        return std::string(_get_name());
    }

    std::string 
    PluginInstance::get_version() const
    {
        return std::string(_get_version());
    }

    std::string
    PluginInstance::get_super_name() const
    {
        return std::string(_get_super_name());
    }

    void 
    PluginManager::loadPluginImpl(const std::string& name)
    {
        std::shared_ptr<PluginInstance> plugin = std::make_shared<PluginInstance>(name);
        _plugins.insert(std::make_pair(plugin->get_name(), plugin));
    }

    std::shared_ptr<PluginInstance> 
    PluginManager::getPluginImpl(const std::string& name)
    {
        return _plugins[name];
    }

    void 
    PluginManager::unloadPluginImpl(const std::string& name)
    {
        _plugins.erase(name);
    }

    void 
    PluginManager::destroyImpl()
    {
        _plugins.clear();
        delete _instance;
    }

} //namespace cupbr