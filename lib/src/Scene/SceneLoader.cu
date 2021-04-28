#include <Scene/SceneLoader.cuh>
#include <Core/Memory.cuh>
#include <Geometry/Plane.cuh>
#include <Geometry/Sphere.cuh>
#include <tinyxml2.h>
#include <vector>
#include <sstream>

namespace detail
{
    Vector3float
    string2vector(const char* str)
    {
        std::stringstream ss(str);
        std::string item;
        std::vector<float> result;

        while(std::getline(ss, item, ','))
        {
            result.push_back(std::stof(item));
        }

        return Vector3float(result[0], result[1], result[2]);
    }

    Material
    loadMaterial(tinyxml2::XMLElement* material_ptr)
    {
        Material material;

        const char* type = material_ptr->FirstChildElement("type")->GetText();
        
        if(strcmp(type, "LAMBERT") == 0)
        {
            material.type = LAMBERT;
        }
        else if(strcmp(type, "PHONG") == 0)
        {
            material.type = PHONG;
        }
        else if(strcmp(type, "MIRROR") == 0)
        {
            material.type = MIRROR;
        }
        else if(strcmp(type, "GLASS") == 0)
        {
            material.type = GLASS;
        }
        else
        {
            std::cerr << "Error while loading material: " << type << " is not a valid material type\n"; 
        }

        //Load material properties
        tinyxml2::XMLElement* albedo_d_string = material_ptr->FirstChildElement("albedo_d");
        tinyxml2::XMLElement* albedo_s_string = material_ptr->FirstChildElement("albedo_s");
        tinyxml2::XMLElement* shininess_string = material_ptr->FirstChildElement("shininess");
        tinyxml2::XMLElement* eta_string = material_ptr->FirstChildElement("eta");

        if(albedo_d_string != NULL)
        {
            material.albedo_d = string2vector(albedo_d_string->GetText());
        }

        if(albedo_s_string != NULL)
        {
            material.albedo_s = string2vector(albedo_s_string->GetText());
        }

        if(shininess_string != NULL)
        {
            material.shininess = std::stof(shininess_string->GetText());
        }

        if(eta_string != NULL)
        {
            material.eta = std::stof(eta_string->GetText());
        }

        return material;
    }
}

Scene
SceneLoader::loadFromFile(const std::string& path)
{
    Scene scene;

    tinyxml2::XMLDocument doc;
    tinyxml2::XMLError error = doc.LoadFile(path.c_str());

    if(error != tinyxml2::XML_SUCCESS)
    {
        std::cerr << "Failed to load XML file: " << path << ". Error code: " << error << "\n";
        return scene;
    }

    //Retrieve information about scene size
    const char* scene_size_string = doc.FirstChildElement("header")->FirstChildElement("scene_size")->GetText();
    const char* light_count_string = doc.FirstChildElement("header")->FirstChildElement("light_count")->GetText();

    uint32_t scene_size = std::stoi(scene_size_string);
    uint32_t light_count = std::stoi(light_count_string);

    scene.scene_size = scene_size;
    scene.light_count = light_count;

    Geometry** host_array = Memory::allocator()->createHostArray<Geometry*>(scene_size);
    Light** host_lights = Memory::allocator()->createHostArray<Light*>(light_count);

    scene.geometry = Memory::allocator()->createDeviceArray<Geometry*>(scene.scene_size);
    scene.lights = Memory::allocator()->createDeviceArray<Light*>(scene.light_count);

    //Load geometry
    tinyxml2::XMLElement* geometry_head = doc.FirstChildElement("geometry");

    for(uint32_t i = 0; i < scene_size; ++i)
    {
        tinyxml2::XMLElement* current_geometry = geometry_head->FirstChildElement(("geometry" + std::to_string(i+1)).c_str());
        const char* type = current_geometry->FirstChildElement("type")->GetText();
        if(strcmp(type, "PLANE") == 0)
        {
            const char* position_string = current_geometry->FirstChildElement("position")->GetText();
            const char* normal_string = current_geometry->FirstChildElement("normal")->GetText();
            Vector3float position = detail::string2vector(position_string);
            Vector3float normal = detail::string2vector(normal_string);
            Material mat = detail::loadMaterial(current_geometry->FirstChildElement("material"));

            Plane geom(position, normal);
            geom.material = mat;

            Plane* dev_geom = Memory::allocator()->createDeviceObject<Plane>();
            Memory::allocator()->copyHost2DeviceObject<Plane>(&geom, dev_geom);
            host_array[i] = dev_geom;
        }
        else if(strcmp(type, "SPHERE") == 0)
        {
            const char* position_string = current_geometry->FirstChildElement("position")->GetText();
            const char* radius_string = current_geometry->FirstChildElement("radius")->GetText();
            Vector3float position = detail::string2vector(position_string);
            float radius = std::stof(radius_string);
            Material mat = detail::loadMaterial(current_geometry->FirstChildElement("material"));

            Sphere geom(position, radius);
            geom.material = mat;
            Sphere* dev_geom = Memory::allocator()->createDeviceObject<Sphere>();
            Memory::allocator()->copyHost2DeviceObject<Sphere>(&geom, dev_geom);
            host_array[i] = dev_geom;
        }
        else
        {
            std::cerr << "Error while loading scene: " << type << " is not a valid geometry type!" << std::endl;
            return scene;
        }

    }

    tinyxml2::XMLElement* light_head = doc.FirstChildElement("lights");
    for(uint32_t i = 0; i < light_count; ++i)
    {
        tinyxml2::XMLElement* current_light = light_head->FirstChildElement(("light" + std::to_string(i+1)).c_str());
        const char* type = current_light->FirstChildElement("type")->GetText();

        Light light;

        if(strcmp(type, "POINT") == 0)
        {
            light.type = POINT;
        }
        else if(strcmp(type, "AREA") == 0)
        {
            light.type = AREA;
        }
        else
        {
            std::cerr << "Error while loading scene: " << type << " is not a valid light type!" << std::endl;
            return scene;
        }

        tinyxml2::XMLElement* position_string = current_light->FirstChildElement("position");
        tinyxml2::XMLElement* intensity_string = current_light->FirstChildElement("intensity");
        tinyxml2::XMLElement* radiance_string = current_light->FirstChildElement("radiance");
        tinyxml2::XMLElement* extend1_string = current_light->FirstChildElement("extend1");
        tinyxml2::XMLElement* extend2_string = current_light->FirstChildElement("extend2");

        if(position_string != NULL)
        {
            light.position = detail::string2vector(position_string->GetText());
        }

        if(intensity_string != NULL)
        {
            light.intensity = detail::string2vector(intensity_string->GetText());
        }

        if(radiance_string != NULL)
        { 
            light.radiance = detail::string2vector(radiance_string->GetText());
        }

        if(extend1_string != NULL)
        {
            light.halfExtend1 = detail::string2vector(extend1_string->GetText());
        }

        if(extend2_string != NULL)
        {
            light.halfExtend2 = detail::string2vector(extend2_string->GetText());
        }

        Light *dev_light = Memory::allocator()->createDeviceObject<Light>();
        Memory::allocator()->copyHost2DeviceObject<Light>(&light, dev_light);

        host_lights[i] = dev_light;
    }

    Memory::allocator()->copyHost2DeviceArray<Geometry*>(scene.scene_size, host_array, scene.geometry);
    //Memory::allocator()->copyHost2DeviceArray<Light*>(scene.light_count, host_lights, scene.lights);

    Memory::allocator()->destroyHostArray<Geometry*>(host_array);
    Memory::allocator()->destroyHostArray<Light*>(host_lights);

    return scene;
}

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

Scene
SceneLoader::cornellBoxSphereAreaLight()
{
    Scene scene;
    scene.scene_size = 8;
    scene.light_count = 2;
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

    Light h_light1, h_light2;
    h_light1.type = AREA;
    h_light1.position = Vector3float(0.0f, 0.9f, 2.0f);
    h_light1.intensity = 1;
    h_light1.radiance = 10;
    h_light1.halfExtend1 = Vector3float(0.1f, 0.0f, 0.0f);
    h_light1.halfExtend2 = Vector3float(0.0f, 0.0f, 0.5f);

    h_light2.type = AREA;
    h_light2.position = Vector3float(0.0f, 0.0f, 2.5f);
    h_light2.intensity = 1;
    h_light2.radiance = Vector3float(0,0,5);
    h_light2.halfExtend1 = Vector3float(0.3f, 0.0f, 0.0f);
    h_light2.halfExtend2 = Vector3float(0.0f, 0.3f, 0.0f);

    Plane* floor = Memory::allocator()->createDeviceObject<Plane>();
    Plane* ceil = Memory::allocator()->createDeviceObject<Plane>();
    Plane* left = Memory::allocator()->createDeviceObject<Plane>();
    Plane* right = Memory::allocator()->createDeviceObject<Plane>();
    Plane* back = Memory::allocator()->createDeviceObject<Plane>();

    Sphere* diff = Memory::allocator()->createDeviceObject<Sphere>();
    Sphere* mirror = Memory::allocator()->createDeviceObject<Sphere>();
    Sphere* glass = Memory::allocator()->createDeviceObject<Sphere>();

    Light* light1 = Memory::allocator()->createDeviceObject<Light>();
    Light* light2 = Memory::allocator()->createDeviceObject<Light>();

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

    Memory::allocator()->copyHost2DeviceObject<Light>(&h_light1, light1);
    Memory::allocator()->copyHost2DeviceObject<Light>(&h_light2, light2);

    Geometry* host_array[] = {floor, ceil, left, right, back, diff, mirror, glass};
    Light* host_lights[] = {light1, light2};

    Memory::allocator()->copyHost2DeviceArray<Geometry*>(scene.scene_size, host_array, scene.geometry);
    Memory::allocator()->copyHost2DeviceArray<Light*>(scene.light_count, host_lights, scene.lights);

    return scene;
}

void
SceneLoader::destroyCornellBoxSphere(Scene scene)
{
    Geometry* host_scene[8];
    Light* host_lights[3];
    Memory::allocator()->copyDevice2HostArray(8, scene.geometry, host_scene);
    Memory::allocator()->copyDevice2HostArray(scene.light_count, scene.lights, host_lights);

    for(uint32_t i = 0; i < 5; ++i)
    {
        Memory::allocator()->destroyDeviceObject<Plane>(static_cast<Plane*>(host_scene[i]));
    }

    for(uint32_t i = 5; i < 8; ++i)
    {
        Memory::allocator()->destroyDeviceObject<Sphere>(static_cast<Sphere*>(host_scene[i]));
    }

    for(uint32_t i = 0; i < scene.light_count; ++i)
    {
        Memory::allocator()->destroyDeviceObject<Light>(host_lights[i]);
    }

    Memory::allocator()->destroyDeviceArray<Geometry*>(scene.geometry);
    Memory::allocator()->destroyDeviceArray<Light*>(scene.lights);
}

Scene
SceneLoader::cornellBoxSphereMultiLight()
{
    Scene scene;
    scene.scene_size = 8;
    scene.light_count = 3;
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

    Light h_light1, h_light2, h_light3;
    h_light1.position = Vector3float(-0.5f, 0.5f, 1.0f);
    h_light1.intensity = Vector3float(1,0,0);

    h_light2.position = Vector3float(0.0f, 0.5f, 1.0f);
    h_light2.intensity = Vector3float(0, 1, 0);

    h_light3.position = Vector3float(0.5f, 0.5f, 1.0f);
    h_light3.intensity = Vector3float(0, 0, 1);

    Plane* floor = Memory::allocator()->createDeviceObject<Plane>();
    Plane* ceil = Memory::allocator()->createDeviceObject<Plane>();
    Plane* left = Memory::allocator()->createDeviceObject<Plane>();
    Plane* right = Memory::allocator()->createDeviceObject<Plane>();
    Plane* back = Memory::allocator()->createDeviceObject<Plane>();

    Sphere* diff = Memory::allocator()->createDeviceObject<Sphere>();
    Sphere* mirror = Memory::allocator()->createDeviceObject<Sphere>();
    Sphere* glass = Memory::allocator()->createDeviceObject<Sphere>();

    Light* light1 = Memory::allocator()->createDeviceObject<Light>();
    Light* light2 = Memory::allocator()->createDeviceObject<Light>();
    Light* light3 = Memory::allocator()->createDeviceObject<Light>();

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

    Memory::allocator()->copyHost2DeviceObject<Light>(&h_light1, light1);
    Memory::allocator()->copyHost2DeviceObject<Light>(&h_light2, light2);
    Memory::allocator()->copyHost2DeviceObject<Light>(&h_light3, light3);

    Geometry* host_array[] = {floor, ceil, left, right, back, diff, mirror, glass};
    Light* host_lights[] = {light1, light2, light3};

    Memory::allocator()->copyHost2DeviceArray<Geometry*>(scene.scene_size, host_array, scene.geometry);
    Memory::allocator()->copyHost2DeviceArray<Light*>(scene.light_count, host_lights, scene.lights);

    return scene;
}