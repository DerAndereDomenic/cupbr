#include <Core/Plugin.h>

#ifdef CUPBR_WINDOWS
#include <Windows.h>
#endif

namespace cupbr
{
    
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
        free(_handle);
    }

    Plugin* 
    PluginInstance::load(const Properties& properties)
    {
        if (!_instance)
            _instance = _load(properties);
        return _instance;
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

} //namespace cupbr