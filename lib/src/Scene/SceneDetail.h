#ifndef __CUPBR_SCENE_SCENEDETAIL_H
#define __CUPBR_SCENE_SCENEDETAIL_H

namespace cupbr
{
    __host__ __device__
    inline Geometry*
    Scene::operator[](const uint32_t index) const
    {
        return geometry[index];
    }
} //namespace cupbr

#endif