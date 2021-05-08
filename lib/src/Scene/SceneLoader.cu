#include <Scene/SceneLoader.cuh>
#include <Core/Memory.cuh>
#include <Geometry/Plane.cuh>
#include <Geometry/Sphere.cuh>
#include <Geometry/Quad.cuh>
#include <tinyxml2.h>
#include <vector>
#include <sstream>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

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
    Scene scene = Scene();

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
        else if(strcmp(type, "QUAD") == 0)
        {
            const char* position_string = current_geometry->FirstChildElement("position")->GetText();
            const char* normal_string = current_geometry->FirstChildElement("normal")->GetText();
            const char* extend1_string = current_geometry->FirstChildElement("extend1")->GetText();
            const char* extend2_string = current_geometry->FirstChildElement("extend2")->GetText();

            Vector3float position = detail::string2vector(position_string);
            Vector3float normal = detail::string2vector(normal_string);
            Vector3float extend1 = detail::string2vector(extend1_string);
            Vector3float extend2 = detail::string2vector(extend2_string);

            Material mat = detail::loadMaterial(current_geometry->FirstChildElement("material"));

            Quad geom(position, normal, extend1, extend2);
            geom.material = mat;

            Quad* dev_geom = Memory::allocator()->createDeviceObject<Quad>();
            Memory::allocator()->copyHost2DeviceObject<Quad>(&geom, dev_geom);
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

    tinyxml2::XMLElement* environment_head = doc.FirstChildElement("environment");
    if(environment_head != NULL)
    {
        const char* path = environment_head->FirstChildElement("path")->GetText();
        
        int32_t x,y,n;
        float *data = stbi_loadf(path, &x, &y, &n, 3);
        Vector3float *img_data = (Vector3float*)data;
        Image<Vector3float> buffer = Image<Vector3float>::createHostObject(x,y);
        Image<Vector3float> dbuffer = Image<Vector3float>::createDeviceObject(x,y);
        for(int i = 0; i < x*y; ++i)
        {
            buffer[i] = img_data[i];
        }
        buffer.copyHost2DeviceObject(dbuffer);

        scene.useEnvironmentMap = true;
        scene.environment = dbuffer;

        Image<Vector3float>::destroyHostObject(buffer);
    }

    Memory::allocator()->copyHost2DeviceArray<Geometry*>(scene.scene_size, host_array, scene.geometry);
    Memory::allocator()->copyHost2DeviceArray<Light*>(scene.light_count, host_lights, scene.lights);

    Memory::allocator()->destroyHostArray<Geometry*>(host_array);
    Memory::allocator()->destroyHostArray<Light*>(host_lights);

    return scene;
}

void
SceneLoader::destroyScene(Scene scene)
{
    Geometry** host_scene = Memory::allocator()->createHostArray<Geometry*>(scene.scene_size);
    Light** host_lights = Memory::allocator()->createHostArray<Light*>(scene.light_count);
    Memory::allocator()->copyDevice2HostArray(scene.scene_size, scene.geometry, host_scene);
    Memory::allocator()->copyDevice2HostArray(scene.light_count, scene.lights, host_lights);

    for(uint32_t i = 0; i < scene.scene_size; ++i)
    {
        Geometry geom;
        Memory::allocator()->copyDevice2HostObject(host_scene[i], &geom);
        switch(geom.type)
        {
            case PLANE:
            {
                Memory::allocator()->destroyDeviceObject<Plane>(static_cast<Plane*>(host_scene[i]));
            }
            break;
            case SPHERE:
            {
                Memory::allocator()->destroyDeviceObject<Sphere>(static_cast<Sphere*>(host_scene[i]));
            }
            break;
        }
    }

    for(uint32_t i = 0; i < scene.light_count; ++i)
    {
        Memory::allocator()->destroyDeviceObject<Light>(host_lights[i]);
    }

    if(scene.useEnvironmentMap)
    {
        Image<Vector3float>::destroyDeviceObject(scene.environment);
    }

    Memory::allocator()->destroyDeviceArray<Geometry*>(scene.geometry);
    Memory::allocator()->destroyDeviceArray<Light*>(scene.lights);
    Memory::allocator()->destroyHostArray<Geometry*>(host_scene);
    Memory::allocator()->destroyHostArray<Light*>(host_lights);
}