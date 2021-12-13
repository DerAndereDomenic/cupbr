#ifndef __CUPBR_DATASTRUCTURE_BOUNDINGVOLUMEHIERARCHY_H
#define __CUPBR_DATASTRUCTURE_BOUNDINGVOLUMEHIERARCHY_H

#include <Geometry/Geometry.h>
#include <vector>

namespace cupbr
{
    /**
    *   @brief Class to model a bounding volume hierarchy
    */
    class BoundingVolumeHierarchy : public Geometry
    {
        public:

        /**
        *   @brief Default constructor
        */
        BoundingVolumeHierarchy() = default;


        BoundingVolumeHierarchy(const std::vector<Geometry*>& host_objects, 
                                const std::vector<Geometry*>& dev_objects);


        void destroy();

        //Override
        __host__ __device__
        LocalGeometry computeRayIntersection(const Ray& ray);

        //private:
        Geometry* _left = nullptr;
        Geometry* _right = nullptr;
        GeometryType _type_left = GeometryType::BVH;
        GeometryType _type_right = GeometryType::BVH;
    };

} //namespace cupbr

#include "../../src/DataStructure/BoundingVolumeHierarchyDetail.h"

#endif