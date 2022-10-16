#ifndef __CUPBR_CORE_TRACINGDETAIL_H
#define __CUPBR_CORE_TRACINGDETAIL_H

#include <Core/CUDA.h>
#include <Core/KernelHelper.h>
#include <Geometry/Plane.h>
#include <Geometry/Sphere.h>
#include <Geometry/Quad.h>
#include <Geometry/Triangle.h>
#include <Geometry/Mesh.h>
#include <Math/Functions.h>

namespace cupbr
{
    CUPBR_DEVICE
    inline Ray
    Tracing::launchRay(const uint32_t& tid, const uint32_t& width, const uint32_t& height, const Camera& camera, const bool& jitter, uint32_t* seed)
    {
        const Vector2uint32_t pixel = ThreadHelper::index2pixel(tid, width, height);

        float jitter_x = jitter ? Math::rnd(*seed) : 0.0f;
        float jitter_y = jitter ? Math::rnd(*seed) : 0.0f;

        const float ratio_x = 2.0f * ((static_cast<float>(pixel.x) + jitter_x) / width - 0.5f);
        const float ratio_y = 2.0f * ((static_cast<float>(pixel.y) + jitter_y) / height - 0.5f);

        const Vector3float world_pos = camera.position() + camera.zAxis() + ratio_x * camera.xAxis() + ratio_y * camera.yAxis();

        return Ray(camera.position(), world_pos - camera.position());
    }

    CUPBR_DEVICE
    inline Ray
    Tracing::launchRay(const Vector2uint32_t& pixel, const uint32_t& width, const uint32_t& height, const Camera& camera, const bool& jitter, uint32_t* seed)
    {
        float jitter_x = jitter ? Math::rnd(*seed) : 0.0f;
        float jitter_y = jitter ? Math::rnd(*seed) : 0.0f;

        const float ratio_x = 2.0f * ((static_cast<float>(pixel.x) + jitter_x) / width - 0.5f);
        const float ratio_y = 2.0f * ((static_cast<float>(pixel.y) + jitter_y) / height - 0.5f);

        const Vector3float world_pos = camera.position() + camera.zAxis() + ratio_x * camera.xAxis() + ratio_y * camera.yAxis();

        return Ray(camera.position(), world_pos - camera.position());
    }

    CUPBR_DEVICE
    inline LocalGeometry
    Tracing::traceRay(GeometryScene& scene, const Ray& ray)
    {
        /*LocalGeometry geom;

        for (uint32_t i = 0; i < scene.scene_size; ++i)
        {
            LocalGeometry curr_geom = Tracing::traceRay(scene, ray, i);

            if(curr_geom.depth < geom.depth)
            {
                geom = curr_geom;
            }
        }

        return geom;*/
        LocalGeometry geom = scene.bvh->computeRayIntersection(ray);
        return geom;
    }

    CUPBR_DEVICE
    inline LocalGeometry
    Tracing::traceRay(GeometryScene& scene, const Ray& ray, const uint32_t& index)
    {
        LocalGeometry geom;

        Geometry* scene_element = scene[index];
        switch (scene_element->type)
        {
            case GeometryType::PLANE:
            {
                Plane* plane = static_cast<Plane*>(scene_element);
                return plane->computeRayIntersection(ray);
            }
            break;
            case GeometryType::SPHERE:
            {
                Sphere* sphere = static_cast<Sphere*>(scene_element);
                return sphere->computeRayIntersection(ray);
            }
            break;
            case GeometryType::QUAD:
            {
                Quad* quad = static_cast<Quad*>(scene_element);
                return quad->computeRayIntersection(ray);
            }
            break;
            case GeometryType::TRIANGLE:
            {
                Triangle* triangle = static_cast<Triangle*>(scene_element);
                return triangle->computeRayIntersection(ray);
            }
            break;
            case GeometryType::MESH:
            {
                Mesh* mesh = static_cast<Mesh*>(scene_element);
                return mesh->computeRayIntersection(ray);
            }
            break;
            case GeometryType::BVH:
            {
                BoundingVolumeHierarchy* bvh = static_cast<BoundingVolumeHierarchy*>(scene_element);
                return bvh->computeRayIntersection(ray);
            }
            break;
        }

        return geom;
    }

    CUPBR_DEVICE
    inline bool
    Tracing::traceVisibility(GeometryScene& scene, const float& lightDist, const Ray& ray)
    {
        /*for (uint32_t i = 0; i < scene.scene_size; ++i)
        {
            Geometry* scene_element = scene[i];
            switch (scene_element->type)
            {
                case GeometryType::PLANE:
                {
                    Plane* plane = static_cast<Plane*>(scene[i]);
                    LocalGeometry geom = plane->computeRayIntersection(ray);
                    if (geom.depth != INFINITY && geom.depth < lightDist)
                    {
                        return false;
                    }
                }
                break;
                case GeometryType::SPHERE:
                {
                    Sphere* sphere = static_cast<Sphere*>(scene_element);
                    LocalGeometry geom = sphere->computeRayIntersection(ray);
                    if (geom.depth != INFINITY && geom.depth < lightDist)
                    {
                        return false;
                    }
                }
                break;
                case GeometryType::QUAD:
                {
                    Quad* quad = static_cast<Quad*>(scene[i]);
                    LocalGeometry geom = quad->computeRayIntersection(ray);
                    if (geom.depth != INFINITY && geom.depth < lightDist)
                    {
                        return false;
                    }
                }
                break;
                case GeometryType::TRIANGLE:
                {
                    Triangle* triangle = static_cast<Triangle*>(scene[i]);
                    LocalGeometry geom = triangle->computeRayIntersection(ray);
                    if (geom.depth != INFINITY && geom.depth < lightDist)
                    {
                        return false;
                    }
                }
                break;
                case GeometryType::MESH:
                {
                    Mesh* mesh = static_cast<Mesh*>(scene[i]);
                    LocalGeometry geom = mesh->computeRayIntersection(ray);
                    if (geom.depth != INFINITY && geom.depth < lightDist)
                    {
                        return false;
                    }
                }
                break;
            }
        }
        return true;*/

        LocalGeometry geom = scene.bvh->computeRayIntersection(ray);
        return geom.depth > lightDist || geom.depth == INFINITY;
    }

    CUPBR_DEVICE
    inline Vector2uint32_t
    Tracing::direction2UV(const Vector3float& direction, const uint32_t& width, const uint32_t& height)
    {
        float theta = acos(direction.y) / M_PI;
        float phi = (atan2(direction.z, direction.x) + M_PI) / (2.0f * M_PI);

        uint32_t x = static_cast<uint32_t>((width - 1) * phi);
        uint32_t y = static_cast<uint32_t>((height - 1) * theta);

        return Vector2uint32_t(x, y);
    }

} //namespace cupbr

#endif