#ifndef __CUPBR_GEOMETRY_AABBDETAIL_H
#define __CUPBR_GEOMETRY_AABBDETAIL_H

#include <Math/Functions.h>
#include <Math/Vector.h>

namespace cupbr
{
    __host__ __device__
    inline
    AABB::AABB(const Vector3float& minimum, const Vector3float& maximum)
        :_minimum(minimum),
         _maximum(maximum)
    {
        
    }
    
    __host__ __device__
    inline bool 
    AABB::hit(const Ray& ray) const
    {
        bool hit = true;
        Vector3float min = _minimum;
        Vector3float max = _maximum;
        float t_min = 0;
        float t_max = INFINITY;
        #define LOOP(a)\
        {\
            float invD = 1.0f / (ray.direction()[a] + EPSILON);\
            float t0 = (min[a] - ray.origin()[a]) * invD;\
            float t1 = (max[a] - ray.origin()[a]) * invD;\
            if(invD < 0.0f)\
            {\
                float temp = t0;\
                t0 = t1;\
                t1 = temp;\
            }\
            t_min = t0 > t_min ? t0 : t_min;\
            t_max = t1 < t_max ? t1 : t_max;\
            if (t_max <= t_min)\
                hit = false;\
        }
        LOOP(0);
        LOOP(1);
        LOOP(2);

        #undef LOOP

        return hit;
    }
    
    __host__ __device__
    inline Vector3float 
    AABB::minimum() const
    {
        return _minimum;
    }
    
    __host__ __device__
    inline Vector3float 
    AABB::maximum() const
    {
        return _maximum;
    }
}

#endif