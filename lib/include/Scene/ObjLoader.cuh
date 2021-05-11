#ifndef __CUPBR_SCENE_OBJLOADER_CUH
#define __CUPBR_SCENE_OBJLOADER_CUH

#include <Scene/Scene.cuh>

namespace ObjLoader
{
    Scene
    loadObj(const char* path);
}

#endif