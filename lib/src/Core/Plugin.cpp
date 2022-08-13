#include <Core/Plugin.h>

#ifdef CUPBR_WINDOWS
#include <Windows.h>
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

        _load = (Plugin * (*)(const Properties & properties))GetProcAddress(*_handle, "load");
        _get_name = (char* (*)())GetProcAddress(*_handle, "name");
        _get_version = (char* (*)())GetProcAddress(*_handle, "version");

        std::cout << "Loaded plugin: " << get_name() << "; Version: " << get_version() << std::endl;

        #endif
    }

    PluginInstance::~PluginInstance()
    {
        //TODO: Free dll
        free(_handle);
    }

    Plugin* 
    PluginInstance::load(const Properties& properties)
    {
        return _load(properties);
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