#ifndef __CUPBR_DATASTRUCTURE_FILEWATCHER_H
#define __CUPBR_DATASTRUCTURE_FILEWATCHER_H

#include <memory>
#include <string>

namespace cupbr
{
    /**
    *   @brief A class to model a file watcher
    */
    class FileWatcher
    {
        public:
        /**
        *   @brief Constructor
        *   @param[in] path The path to the file to watch
        */
        FileWatcher(const std::string& path);

        /**
        *   @brief Destructor
        */
        ~FileWatcher();

        /**
        *   @brief Check if a file was updated
        *   @return True if the file was updated since the last check
        */
        bool fileUpdated();

        /**
        *   @brief Set the path of the file to watch
        *   @param[in] path The file path
        */
        void setPath(const std::string& path);

        private:
        class Impl;
        std::unique_ptr<Impl> impl;
    };
} //namespace cupbr

#endif