#ifndef __CUPBR_DATASTRUCTURE_BOUNDINGVOLUMEHIERARCHYDETAIL_H
#define __CUPBR_DATASTRUCTURE_BOUNDINGVOLUMEHIERARCHYDETAIL_H

#include <Geometry/Plane.h>
#include <Geometry/Quad.h>
#include <Geometry/Sphere.h>
#include <Geometry/Mesh.h>

namespace cupbr
{
    CUPBR_HOST_DEVICE
    inline LocalGeometry 
    BoundingVolumeHierarchy::computeRayIntersection(const Ray& ray)
    {
        #define pushStack(x) _stack[stack_size++] = x
        #define popStack() _stack[--stack_size]
        Geometry* _stack[2048]; // Space for (guaranteed) 1000 objects
        uint32_t stack_size = 1;
        Geometry* current_node;
        _stack[0] = this;

        LocalGeometry best;
        while (stack_size != 0)
        {
            current_node = popStack();

            bool geometry_hit = false;
            while(!geometry_hit)
            {
                if (!current_node->aabb().hit(ray))
                {
                    break;
                }

                LocalGeometry geom;
                switch (current_node->type)
                {
                    case GeometryType::PLANE:
                    {
                        Plane* plane = static_cast<Plane*>(current_node);
                        geom = plane->computeRayIntersection(ray);
                        if (geom.depth < best.depth)
                            best = geom;
                        geometry_hit = true;
                    }
                    break;
                    case GeometryType::SPHERE:
                    {
                        Sphere* sphere = static_cast<Sphere*>(current_node);
                        geom = sphere->computeRayIntersection(ray);
                        if (geom.depth < best.depth)
                            best = geom;
                        geometry_hit = true;
                    }
                    break;
                    case GeometryType::QUAD:
                    {
                        Quad* quad = static_cast<Quad*>(current_node);
                        geom = quad->computeRayIntersection(ray);
                        if (geom.depth < best.depth)
                            best = geom;
                        geometry_hit = true;
                    }
                    break;
                    case GeometryType::TRIANGLE:
                    {
                        Triangle* triangle = static_cast<Triangle*>(current_node);
                        geom = triangle->computeRayIntersection(ray);
                        if (geom.depth < best.depth)
                            best = geom;
                        geometry_hit = true;
                    }
                    break;
                    case GeometryType::MESH:
                    {
                        Mesh* mesh = static_cast<Mesh*>(current_node);
                        geom = mesh->computeRayIntersection(ray);
                        if (geom.depth < best.depth)
                            best = geom;
                        geometry_hit = true;
                    }
                    break;
                    case GeometryType::BVH:
                    {
                        BoundingVolumeHierarchy* bvh = static_cast<BoundingVolumeHierarchy*>(current_node);
                        if (bvh->_right != bvh->_left)
                           pushStack(bvh->_right);

                        current_node = bvh->_left;
                    }
                    break;
                }
            }

        } 

        #undef push
        #undef pop

        return best;
    }
}

#endif