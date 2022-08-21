#include <DataStructure/FileWatcher.h>
#include <filesystem>

namespace cupbr
{

    class FileWatcher::Impl
    {
        public:

        Impl() = default;

        ~Impl() = default;

        std::string path;

        std::filesystem::file_time_type last_write_time;
    };

    FileWatcher::FileWatcher(const std::string& path)
    {
        impl = std::make_unique<Impl>();
        setPath(path);
    }

    FileWatcher::~FileWatcher() = default;

    bool 
    FileWatcher::fileUpdated()
    {
        auto current_time = std::filesystem::last_write_time(impl->path);
        if(current_time != impl->last_write_time)
        {
            impl->last_write_time = current_time;
            return true;
        }
        return false;
    }

    void 
    FileWatcher::setPath(const std::string& path)
    {
        impl->path = path;
        impl->last_write_time = std::filesystem::last_write_time(path);
    }

} //namespace cupbr