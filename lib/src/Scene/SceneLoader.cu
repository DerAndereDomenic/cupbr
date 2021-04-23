#include <Scene/SceneLoader.cuh>
#include <Core/Memory.cuh>
#include <Geometry/Plane.cuh>
#include <Geometry/Sphere.cuh>

Scene
SceneLoader::cornellBoxSphere()
{
    Scene scene;
    scene.scene_size = 8;
    scene.light_count = 1;
    scene.geometry = Memory::allocator()->createDeviceArray<Geometry*>(scene.scene_size);
    scene.lights = Memory::allocator()->createDeviceArray<Light*>(scene.light_count);

    Plane h_floor(Vector3float(0,-1,0), Vector3float(0,1,0));
    Plane h_ceil(Vector3float(0,1,0), Vector3float(0,-1,0));
    Plane h_left(Vector3float(-1,0,0), Vector3float(1,0,0));
    Plane h_right(Vector3float(1,0,0), Vector3float(-1,0,0));
    Plane h_back(Vector3float(0,0,3), Vector3float(0,0,-1)); 

    Sphere h_diff(Vector3float(-0.65f,-0.75f,2.65f), 0.25f);
    Sphere h_mirror(Vector3float(0.65f,-0.75f,2.65f), 0.25f);
    Sphere h_glass(Vector3float(0.0f,-0.75f,1.25f), 0.25f);

    Light h_light;
    h_light.position = Vector3float(0.0f, 0.9f, 2.0f);
    h_light.intensity = 1;

    Plane* floor = Memory::allocator()->createDeviceObject<Plane>();
    Plane* ceil = Memory::allocator()->createDeviceObject<Plane>();
    Plane* left = Memory::allocator()->createDeviceObject<Plane>();
    Plane* right = Memory::allocator()->createDeviceObject<Plane>();
    Plane* back = Memory::allocator()->createDeviceObject<Plane>();

    Sphere* diff = Memory::allocator()->createDeviceObject<Sphere>();
    Sphere* mirror = Memory::allocator()->createDeviceObject<Sphere>();
    Sphere* glass = Memory::allocator()->createDeviceObject<Sphere>();

    Light* light = Memory::allocator()->createDeviceObject<Light>();

    h_left.material.albedo_d = Vector3float(0,1,0);
    h_right.material.albedo_d = Vector3float(1,0,0);
    h_diff.material.albedo_d = Vector3float(0,0,1);
    h_diff.material.albedo_s = Vector3float(1,1,1);
    h_diff.material.type = MaterialType::PHONG;
    h_mirror.material.type = MaterialType::MIRROR;
    h_mirror.material.albedo_s = 1;
    h_glass.material.type = MaterialType::GLASS;
    h_glass.material.albedo_s = 1;

    Memory::allocator()->copyHost2DeviceObject<Plane>(&h_floor, floor);
    Memory::allocator()->copyHost2DeviceObject<Plane>(&h_ceil, ceil);
    Memory::allocator()->copyHost2DeviceObject<Plane>(&h_left, left);
    Memory::allocator()->copyHost2DeviceObject<Plane>(&h_right, right);
    Memory::allocator()->copyHost2DeviceObject<Plane>(&h_back, back);

    Memory::allocator()->copyHost2DeviceObject<Sphere>(&h_diff, diff);
    Memory::allocator()->copyHost2DeviceObject<Sphere>(&h_mirror, mirror);
    Memory::allocator()->copyHost2DeviceObject<Sphere>(&h_glass, glass);

    Memory::allocator()->copyHost2DeviceObject<Light>(&h_light, light);

    Geometry* host_array[] = {floor, ceil, left, right, back, diff, mirror, glass};
    Light* host_lights[] = {light};

    Memory::allocator()->copyHost2DeviceArray<Geometry*>(scene.scene_size, host_array, scene.geometry);
    Memory::allocator()->copyHost2DeviceArray<Light*>(scene.light_count, host_lights, scene.lights);

    return scene;
}

void
SceneLoader::destroyCornellBoxSphere(Scene scene)
{
    Geometry* host_scene[8];
    Light* host_lights[1];
    Memory::allocator()->copyDevice2HostArray(8, scene.geometry, host_scene);
    Memory::allocator()->copyDevice2HostArray(1, scene.lights, host_lights);

    for(uint32_t i = 0; i < 5; ++i)
    {
        Memory::allocator()->destroyDeviceObject<Plane>(static_cast<Plane*>(host_scene[i]));
    }

    for(uint32_t i = 5; i < 8; ++i)
    {
        Memory::allocator()->destroyDeviceObject<Sphere>(static_cast<Sphere*>(host_scene[i]));
    }

    for(uint32_t i = 0; i < 1; ++i)
    {
        Memory::allocator()->destroyDeviceObject<Light>(host_lights[i]);
    }

    Memory::allocator()->destroyDeviceArray<Geometry*>(scene.geometry);
    Memory::allocator()->destroyDeviceArray<Light*>(scene.lights);
}