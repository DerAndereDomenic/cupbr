#ifndef __CUPBR_SCENE_OBJLOADER_H
#define __CUPBR_SCENE_OBJLOADER_H

#include <Geometry/Mesh.h>

namespace cupbr
{
    namespace ObjLoader
    {
        /**
        *   @brief Load a mesh from an obj file
        *   @param[in] path The path
        *   @param[in] position An offset for the mesh
        *   @param[in] scale A scale for the mesh
        *   @return A (host) pointer to a mesh
        */
        Mesh* loadObj(const char* path, const Vector3float& position, const Vector3float& scale);

    } //namespace ObjLoader
} //namespace cupbr

#endif