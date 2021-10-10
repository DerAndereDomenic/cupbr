#ifndef __CUPBR_SCENE_OBJLOADER_CUH
#define __CUPBR_SCENE_OBJLOADER_CUH

#include <Geometry/Mesh.cuh>

namespace cupbr
{
    namespace ObjLoader
    {
        /**
        *   @brief Load a mesh from an obj file
        *   @param[in] path The path
        *   @param[in] position An offset for the mesh
        *   @return A (host) pointer to a mesh
        */
        Mesh*
            loadObj(const char* path, const Vector3float& position);
    } //namespace ObjLoader
} //namespace cupbr

#endif