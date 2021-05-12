#ifndef __CUPBR_SCENE_OBJLOADER_CUH
#define __CUPBR_SCENE_OBJLOADER_CUH

#include <Geometry/Mesh.cuh>

namespace ObjLoader
{
    /**
    *   @brief Load a mesh from an obj file
    *   @param[in] path The path
    *   @return A (host) pointer to a mesh 
    */
    Mesh*
    loadObj(const char* path);
}

#endif