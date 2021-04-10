#include <Scene/SceneLoader.cuh>
#include <Core/Memory.cuh>

Scene
SceneLoader::cornellBoxSphere(uint32_t* scene_size)
{
    *scene_size = 8;

    Scene scene = Memory::allocator()->createDeviceArray<Geometry*>(*scene_size);

    Plane h_floor(Vector3float(0,-1,0), Vector3float(0,1,0));
    Plane h_ceil(Vector3float(0,1,0), Vector3float(0,-1,0));
    Plane h_left(Vector3float(-1,0,0), Vector3float(1,0,0));
    Plane h_right(Vector3float(1,0,0), Vector3float(-1,0,0));
    Plane h_back(Vector3float(0,0,3), Vector3float(0,0,-1)); 

    Sphere h_diff(Vector3float(-0.65f,-0.75f,2.65f), 0.25f);
    Sphere h_mirror(Vector3float(0.65f,-0.75f,2.65f), 0.25f);
    Sphere h_glass(Vector3float(0.0f,-0.75f,1.25f), 0.25f);

    Plane* floor = Memory::allocator()->createDeviceObject<Plane>();
    Plane* ceil = Memory::allocator()->createDeviceObject<Plane>();
    Plane* left = Memory::allocator()->createDeviceObject<Plane>();
    Plane* right = Memory::allocator()->createDeviceObject<Plane>();
    Plane* back = Memory::allocator()->createDeviceObject<Plane>();

    Sphere* diff = Memory::allocator()->createDeviceObject<Sphere>();
    Sphere* mirror = Memory::allocator()->createDeviceObject<Sphere>();
    Sphere* glass = Memory::allocator()->createDeviceObject<Sphere>();

    h_left.material.albedo_d = Vector3float(0,1,0);
    h_right.material.albedo_d = Vector3float(1,0,0);

    Memory::allocator()->copyHost2DeviceObject<Plane>(&h_floor, floor);
    Memory::allocator()->copyHost2DeviceObject<Plane>(&h_ceil, ceil);
    Memory::allocator()->copyHost2DeviceObject<Plane>(&h_left, left);
    Memory::allocator()->copyHost2DeviceObject<Plane>(&h_right, right);
    Memory::allocator()->copyHost2DeviceObject<Plane>(&h_back, back);

    Memory::allocator()->copyHost2DeviceObject<Sphere>(&h_diff, diff);
    Memory::allocator()->copyHost2DeviceObject<Sphere>(&h_mirror, mirror);
    Memory::allocator()->copyHost2DeviceObject<Sphere>(&h_glass, glass);

    Geometry* host_array[] = {floor, ceil, left, right, back, diff, mirror, glass};

    Memory::allocator()->copyHost2DeviceObject<Geometry*>(*scene_size, host_array, scene);

    return scene;
}

void
SceneLoader::destroyCornellBoxSphere(Scene scene)
{

}