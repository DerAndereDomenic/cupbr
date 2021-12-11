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
    __device__
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

    __device__
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

    __device__
    inline LocalGeometry
    Tracing::traceRay(Scene& scene, const Ray& ray)
    {
        LocalGeometry geom;

        for (uint32_t i = 0; i < scene.scene_size; ++i)
        {
            LocalGeometry curr_geom = Tracing::traceRay(scene, ray, i);

            if(curr_geom.depth < geom.depth)
            {
                geom = curr_geom;
            }
        }

        return geom;
    }

    __device__
    inline LocalGeometry
    Tracing::traceRay(Scene& scene, const Ray& ray, const uint32_t& index)
    {
        LocalGeometry geom;

        Geometry* scene_element = scene[index];
        switch (scene_element->type)
        {
            case GeometryType::PLANE:
            {
                Plane* plane = static_cast<Plane*>(scene_element);
                Vector4float intersection_plane = plane->computeRayIntersection(ray);
                geom.type = GeometryType::PLANE;
                geom.P = Vector3float(intersection_plane);
                geom.N = plane->getNormal(geom.P);
                geom.depth = intersection_plane.w;
                geom.material = plane->material;
                geom.scene_index = index;
            }
            break;
            case GeometryType::SPHERE:
            {
                Sphere* sphere = static_cast<Sphere*>(scene_element);
                Vector4float intersection_sphere = sphere->computeRayIntersection(ray);
                geom.type = GeometryType::SPHERE;
                geom.P = Vector3float(intersection_sphere);
                geom.N = sphere->getNormal(geom.P);
                geom.depth = intersection_sphere.w;
                geom.material = sphere->material;
                geom.scene_index = index;
            }
            break;
            case GeometryType::QUAD:
            {
                Quad* quad = static_cast<Quad*>(scene_element);
                Vector4float intersection_quad = quad->computeRayIntersection(ray);
                geom.type = GeometryType::PLANE;
                geom.P = Vector3float(intersection_quad);
                geom.N = quad->getNormal(geom.P);
                geom.depth = intersection_quad.w;
                geom.material = quad->material;
                geom.scene_index = index;
            }
            break;
            case GeometryType::TRIANGLE:
            {
                Triangle* triangle = static_cast<Triangle*>(scene_element);
                Vector4float intersection_triangle = triangle->computeRayIntersection(ray);
                geom.type = GeometryType::TRIANGLE;
                geom.P = Vector3float(intersection_triangle);
                geom.N = triangle->getNormal(geom.P);
                geom.depth = intersection_triangle.w;
                geom.material = triangle->material;
                geom.scene_index = index;
            }
            break;
            case GeometryType::MESH:
            {
                Mesh* mesh = static_cast<Mesh*>(scene_element);
                Vector4float intersection_mesh = mesh->computeRayIntersection(ray);
                geom.type = GeometryType::MESH;
                geom.P = Vector3float(intersection_mesh);
                geom.N = mesh->getNormal(geom.P);
                geom.depth = intersection_mesh.w;
                geom.material = mesh->material;
                geom.scene_index = index;
            }
            break;
        }

        return geom;
    }

    __device__
    inline bool
    Tracing::traceVisibility(Scene& scene, const float& lightDist, const Ray& ray)
    {
        for (uint32_t i = 0; i < scene.scene_size; ++i)
        {
            Geometry* scene_element = scene[i];
            switch (scene_element->type)
            {
                case GeometryType::PLANE:
                {
                    Plane* plane = static_cast<Plane*>(scene[i]);
                    Vector4float intersection_plane = plane->computeRayIntersection(ray);
                    if (intersection_plane.w != INFINITY && intersection_plane.w < lightDist)
                    {
                        return false;
                    }
                }
                break;
                case GeometryType::SPHERE:
                {
                    Sphere* sphere = static_cast<Sphere*>(scene_element);
                    Vector4float intersection_sphere = sphere->computeRayIntersection(ray);
                    if (intersection_sphere.w != INFINITY && intersection_sphere.w < lightDist)
                    {
                        return false;
                    }
                }
                break;
                case GeometryType::QUAD:
                {
                    Quad* quad = static_cast<Quad*>(scene[i]);
                    Vector4float intersection_quad = quad->computeRayIntersection(ray);
                    if (intersection_quad.w != INFINITY && intersection_quad.w < lightDist)
                    {
                        return false;
                    }
                }
                break;
                case GeometryType::TRIANGLE:
                {
                    Triangle* triangle = static_cast<Triangle*>(scene[i]);
                    Vector4float intersection_triangle = triangle->computeRayIntersection(ray);
                    if (intersection_triangle.w != INFINITY && intersection_triangle.w < lightDist)
                    {
                        return false;
                    }
                }
                break;
                case GeometryType::MESH:
                {
                    Mesh* mesh = static_cast<Mesh*>(scene[i]);
                    Vector4float intersection_mesh = mesh->computeRayIntersection(ray);
                    if (intersection_mesh.w != INFINITY && intersection_mesh.w < lightDist)
                    {
                        return false;
                    }
                }
                break;
            }
        }
        return true;
    }

    __device__
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