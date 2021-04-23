#ifndef __CUPBR_CORE_TRACINGDETAIL_CUH
#define __CUPBR_CORE_TRACINGDETAIL_CUH

#include <Core/CUDA.cuh>
#include <Core/KernelHelper.cuh>
#include <Geometry/Plane.cuh>
#include <Geometry/Sphere.cuh>

__device__
inline Ray
Tracing::launchRay(const uint32_t& tid, const uint32_t& width, const uint32_t& height, const Camera& camera)
{
    const Vector2uint32_t pixel = ThreadHelper::index2pixel(tid, width, height);

    const float ratio_x = 2.0f*(static_cast<float>(pixel.x)/width - 0.5f);
    const float ratio_y = 2.0f*(static_cast<float>(pixel.y)/height - 0.5f);

    const Vector3float world_pos = camera.position() + camera.zAxis() + ratio_x*camera.xAxis() + ratio_y*camera.yAxis();

    return Ray(camera.position(), world_pos - camera.position());
}

__device__
inline LocalGeometry
Tracing::traceRay(const Scene scene, const Ray& ray)
{
    Vector4float intersection(INFINITY);

    LocalGeometry geom;

    for(uint32_t i = 0; i < scene.scene_size; ++i)
    {
        Geometry* scene_element = scene[i];
        switch(scene_element->type)
        {
            case GeometryType::PLANE:
            {
                Plane *plane = static_cast<Plane*>(scene[i]);
                Vector4float intersection_plane = plane->computeRayIntersection(ray);
                if(intersection_plane.w < intersection.w)
                {
                    intersection = intersection_plane;
                    geom.type = GeometryType::PLANE;
                    geom.P = Vector3float(intersection);
                    geom.N = plane->getNormal(geom.P);
                    geom.depth = intersection.w;
                    geom.material = plane->material;
                }
            }
            break;
            case GeometryType::SPHERE:
            {
                Sphere *sphere = static_cast<Sphere*>(scene_element);
                Vector4float intersection_sphere = sphere->computeRayIntersection(ray);
                if(intersection_sphere.w < intersection.w)
                {
                    intersection = intersection_sphere;
                    geom.type = GeometryType::SPHERE;
                    geom.P = Vector3float(intersection);
                    geom.N = sphere->getNormal(geom.P);
                    geom.depth = intersection.w;
                    geom.material = sphere->material;
                }
            }
            break;
        }
    }

    return geom;
}

__device__
inline bool
Tracing::traceVisibility(const Scene scene, const float& lightDist, const Ray& ray)
{
    for(uint32_t i = 0; i < scene.scene_size; ++i)
    {
        Geometry* scene_element = scene[i];
        switch(scene_element->type)
        {
            case GeometryType::PLANE:
            {
                Plane *plane = static_cast<Plane*>(scene[i]);
                Vector4float intersection_plane = plane->computeRayIntersection(ray);
                if(intersection_plane.w != INFINITY && intersection_plane.w < lightDist)
                {
                    return false;
                }
            }
            break;
            case GeometryType::SPHERE:
            {
                Sphere *sphere = static_cast<Sphere*>(scene_element);
                Vector4float intersection_sphere = sphere->computeRayIntersection(ray);
                if(intersection_sphere.w != INFINITY && intersection_sphere.w < lightDist)
                {
                    return false;
                }
            }
            break;
        }
    }
    return true;
}

#endif