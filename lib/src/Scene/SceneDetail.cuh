#ifndef __CUPBR_SCENE_SCENEDETAIL_CUH
#define __CUPBR_SCENE_SCENEDETAIL_CUH

__host__ __device__
inline Geometry* 
Scene::operator[](const uint32_t index) const
{
    return geometry[index];
}

#endif