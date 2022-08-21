#ifndef __CUPBR_CORE_ENTRYPOINT_H
#define __CUPBR_CORE_ENTRYPOINT_H

#include <CUPBR.h>

extern int run(int argc, char* argv[]);

int main(int argc, char* argv[])
{
    setDefaultDevice();

    std::vector<std::filesystem::path> paths;
    for (auto entry : std::filesystem::directory_iterator(CUPBR_PLUGIN_PATH))
    {
        paths.push_back(entry.path());
    }

    for (auto s : paths)
    {
        if(s.filename().extension().string() == CUPBR_PLUGIN_FILE_ENDING)
        {
            cupbr::PluginManager::loadPlugin(s.filename().stem().string());
        }
    }

    int exit =  run(argc, argv);
    cupbr::Memory::printStatistics();
    return exit;
}

#endif