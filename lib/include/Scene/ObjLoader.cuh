#ifndef __CUPBR_SCENE_OBJLOADER_CUH
#define __CUPBR_SCENE_OBJLOADER_CUH

#include <Geometry/Mesh.cuh>

namespace ObjLoader
{
    Mesh*
    loadObj(const char* path);
}

#endif