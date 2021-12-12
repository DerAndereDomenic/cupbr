#include <Scene/SceneLoader.h>
#include <Core/Memory.h>
#include <Geometry/Plane.h>
#include <Geometry/Sphere.h>
#include <Geometry/Quad.h>
#include <Geometry/Triangle.h>
#include <Geometry/Mesh.h>
#include <Scene/ObjLoader.h>
#include <tinyxml2.h>
#include <vector>
#include <sstream>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace cupbr
{
    namespace detail
    {
        Vector3float
        string2vector(const char* str)
        {
            std::stringstream ss(str);
            std::string item;
            std::vector<float> result;

            while (std::getline(ss, item, ','))
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

            if (strcmp(type, "LAMBERT") == 0)
            {
                material.type = MaterialType::LAMBERT;
            }
            else if (strcmp(type, "PHONG") == 0)
            {
                material.type = MaterialType::PHONG;
            }
            else if (strcmp(type, "MIRROR") == 0)
            {
                material.type = MaterialType::MIRROR;
            }
            else if (strcmp(type, "GLASS") == 0)
            {
                material.type = MaterialType::GLASS;
            }
            else if (strcmp(type, "GGX") == 0)
            {
                material.type = MaterialType::GGX;
            }
            else if(strcmp(type, "VOLUME") == 0)
            {
                material.type = MaterialType::VOLUME;
            }
            else
            {
                std::cerr << "Error while loading material: " << type << " is not a valid material type\n";
            }

            //Load material properties
            tinyxml2::XMLElement* albedo_e_string = material_ptr->FirstChildElement("albedo_e");
            tinyxml2::XMLElement* albedo_d_string = material_ptr->FirstChildElement("albedo_d");
            tinyxml2::XMLElement* albedo_s_string = material_ptr->FirstChildElement("albedo_s");
            tinyxml2::XMLElement* shininess_string = material_ptr->FirstChildElement("shininess");
            tinyxml2::XMLElement* roughness_string = material_ptr->FirstChildElement("roughness");
            tinyxml2::XMLElement* eta_string = material_ptr->FirstChildElement("eta");
            tinyxml2::XMLElement* sigma_a_string = material_ptr->FirstChildElement("sigma_a");
            tinyxml2::XMLElement* sigma_s_string = material_ptr->FirstChildElement("sigma_s");
            tinyxml2::XMLElement* g_string = material_ptr->FirstChildElement("g");

            if (albedo_e_string != NULL)
            {
                material.albedo_e = string2vector(albedo_e_string->GetText());
            }

            if (albedo_d_string != NULL)
            {
                material.albedo_d = string2vector(albedo_d_string->GetText());
            }

            if (albedo_s_string != NULL)
            {
                material.albedo_s = string2vector(albedo_s_string->GetText());
            }

            if (shininess_string != NULL)
            {
                material.shininess = std::stof(shininess_string->GetText());
            }

            if (roughness_string != NULL)
            {
                material.roughness = std::max(1e-4f, std::stof(roughness_string->GetText()));
            }

            if (eta_string != NULL)
            {
                material.eta = std::stof(eta_string->GetText());
            }

            if(sigma_s_string != NULL)
            {
                material.volume.sigma_s = string2vector(sigma_s_string->GetText());
            }

            if(sigma_a_string != NULL)
            {
                material.volume.sigma_a = string2vector(sigma_a_string->GetText());
            }

            if(g_string != NULL)
            {
                material.volume.g = std::stof(g_string->GetText());
            }

            return material;
        }
    } //namespace detail

    Scene
    SceneLoader::loadFromFile(const std::string& path)
    {
        Scene scene = Scene();

        tinyxml2::XMLDocument doc;
        tinyxml2::XMLError error = doc.LoadFile(path.c_str());

        if (error != tinyxml2::XML_SUCCESS)
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

        std::vector<Geometry*> host_geometry;
        std::vector<Light*> host_lights;

        scene.geometry = Memory::createDeviceArray<Geometry*>(scene.scene_size);
        scene.lights = Memory::createDeviceArray<Light*>(scene.light_count);

        //Load geometry
        tinyxml2::XMLElement* geometry_head = doc.FirstChildElement("geometry");

        for (uint32_t i = 0; i < scene_size; ++i)
        {
            tinyxml2::XMLElement* current_geometry = geometry_head->FirstChildElement(("geometry" + std::to_string(i + 1)).c_str());
            const char* type = current_geometry->FirstChildElement("type")->GetText();
            if (strcmp(type, "PLANE") == 0)
            {
                const char* position_string = current_geometry->FirstChildElement("position")->GetText();
                const char* normal_string = current_geometry->FirstChildElement("normal")->GetText();
                Vector3float position = detail::string2vector(position_string);
                Vector3float normal = detail::string2vector(normal_string);
                Material mat = detail::loadMaterial(current_geometry->FirstChildElement("material"));

                Plane* geom = new Plane(position, normal);
                geom->material = mat;

                host_geometry.push_back(geom);
            }
            else if (strcmp(type, "SPHERE") == 0)
            {
                const char* position_string = current_geometry->FirstChildElement("position")->GetText();
                const char* radius_string = current_geometry->FirstChildElement("radius")->GetText();
                Vector3float position = detail::string2vector(position_string);
                float radius = std::stof(radius_string);
                Material mat = detail::loadMaterial(current_geometry->FirstChildElement("material"));

                Sphere* geom = new Sphere(position, radius);
                geom->material = mat;
                host_geometry.push_back(geom);
            }
            else if (strcmp(type, "QUAD") == 0)
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

                Quad* geom = new Quad(position, normal, extend1, extend2);
                geom->material = mat;

                host_geometry.push_back(geom);
            }
            else if (strcmp(type, "TRIANGLE") == 0)
            {
                const char* vertex1_string = current_geometry->FirstChildElement("vertex1")->GetText();
                const char* vertex2_string = current_geometry->FirstChildElement("vertex2")->GetText();
                const char* vertex3_string = current_geometry->FirstChildElement("vertex3")->GetText();

                Vector3float vertex1 = detail::string2vector(vertex1_string);
                Vector3float vertex2 = detail::string2vector(vertex2_string);
                Vector3float vertex3 = detail::string2vector(vertex3_string);

                Material mat = detail::loadMaterial(current_geometry->FirstChildElement("material"));

                Triangle* geom = new Triangle(vertex1, vertex2, vertex3);
                geom->material = mat;

                host_geometry.push_back(geom);
            }
            else if (strcmp(type, "MESH") == 0)
            {
                const char* path_string = current_geometry->FirstChildElement("path")->GetText();
                const char* position_string = current_geometry->FirstChildElement("position")->GetText();

                Vector3float position = detail::string2vector(position_string);

                Mesh* geom = ObjLoader::loadObj(path_string, position);

                Material mat = detail::loadMaterial(current_geometry->FirstChildElement("material"));

                geom->material = mat;

                host_geometry.push_back(geom);
            }
            else
            {
                std::cerr << "Error while loading scene: " << type << " is not a valid geometry type!" << std::endl;
                return scene;
            }

        }

        tinyxml2::XMLElement* light_head = doc.FirstChildElement("lights");
        for (uint32_t i = 0; i < light_count; ++i)
        {
            tinyxml2::XMLElement* current_light = light_head->FirstChildElement(("light" + std::to_string(i + 1)).c_str());
            const char* type = current_light->FirstChildElement("type")->GetText();

            Light light;

            if (strcmp(type, "POINT") == 0)
            {
                light.type = POINT;
            }
            else if (strcmp(type, "AREA") == 0)
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

            if (position_string != NULL)
            {
                light.position = detail::string2vector(position_string->GetText());
            }

            if (intensity_string != NULL)
            {
                light.intensity = detail::string2vector(intensity_string->GetText());
            }

            if (radiance_string != NULL)
            {
                light.radiance = detail::string2vector(radiance_string->GetText());
            }

            if (extend1_string != NULL)
            {
                light.halfExtend1 = detail::string2vector(extend1_string->GetText());
            }

            if (extend2_string != NULL)
            {
                light.halfExtend2 = detail::string2vector(extend2_string->GetText());
            }

            Light* light_ptr = new Light(light);

            host_lights.push_back(light_ptr);
        }

        tinyxml2::XMLElement* volume_head = doc.FirstChildElement("volume");
        if (volume_head != NULL)
        {
            const char* sigma_s_string = volume_head->FirstChildElement("sigma_s")->GetText();
            const char* sigma_a_string = volume_head->FirstChildElement("sigma_a")->GetText();
            const char* g_string = volume_head->FirstChildElement("g")->GetText();

            Volume vol;

            vol.sigma_s = detail::string2vector(sigma_s_string);
            vol.sigma_a = detail::string2vector(sigma_a_string);
            vol.g = std::stof(g_string);

            scene.volume = vol;
        }

        tinyxml2::XMLElement* environment_head = doc.FirstChildElement("environment");
        if (environment_head != NULL)
        {
            const char* path = environment_head->FirstChildElement("path")->GetText();

            int32_t x, y, n;
            float* data = stbi_loadf(path, &x, &y, &n, 3);
            Vector3float* img_data = (Vector3float*)data;
            Image<Vector3float> buffer = Image<Vector3float>::createHostObject(x, y);
            Image<Vector3float> dbuffer = Image<Vector3float>::createDeviceObject(x, y);
            for (int i = 0; i < x * y; ++i)
            {
                buffer[i] = img_data[i];
            }
            buffer.copyHost2DeviceObject(dbuffer);

            scene.useEnvironmentMap = true;
            scene.environment = dbuffer;

            stbi_image_free(data);

            Image<Vector3float>::destroyHostObject(buffer);
        }

        //Transfer data to device
        //TODO: Backend
        std::vector<Geometry*> dev_geometry;
        for(uint32_t i = 0; i < host_geometry.size(); ++i)
        {
            Geometry* geom = host_geometry[i];
            switch(geom->type)
            {
                case GeometryType::SPHERE:
                {
                    Sphere* dev_geom = Memory::createDeviceObject<Sphere>();
                    Memory::copyHost2DeviceObject<Sphere>(static_cast<Sphere*>(geom), dev_geom);
                    dev_geometry.push_back(dev_geom);
                    delete geom;
                }
                break;
                case GeometryType::PLANE:
                {
                    Plane* dev_geom = Memory::createDeviceObject<Plane>();
                    Memory::copyHost2DeviceObject<Plane>(static_cast<Plane*>(geom), dev_geom);
                    dev_geometry.push_back(dev_geom);
                    delete geom;
                }
                break;
                case GeometryType::QUAD:
                {
                    Quad* dev_geom = Memory::createDeviceObject<Quad>();
                    Memory::copyHost2DeviceObject<Quad>(static_cast<Quad*>(geom), dev_geom);
                    dev_geometry.push_back(dev_geom);
                    delete geom;
                }
                break;
                case GeometryType::TRIANGLE:
                {
                    Triangle* dev_geom = Memory::createDeviceObject<Triangle>();
                    Memory::copyHost2DeviceObject<Triangle>(static_cast<Triangle*>(geom), dev_geom);
                    dev_geometry.push_back(dev_geom);
                    delete geom;
                }
                break;
                case GeometryType::MESH:
                {
                    Mesh* dev_geom = Memory::createDeviceObject<Mesh>();
                    Memory::copyHost2DeviceObject<Mesh>(static_cast<Mesh*>(geom), dev_geom);
                    dev_geometry.push_back(dev_geom);
                    Memory::destroyHostObject<Mesh>(static_cast<Mesh*>(geom));
                }
                break;
            }
        }

        Memory::copyHost2DeviceArray(host_geometry.size(), dev_geometry.data(), scene.geometry);

        std::vector<Light*> dev_lights;
        for(uint32_t i = 0; i < host_lights.size(); ++i)
        {
            Light* light = host_lights[i];
            Light* dev_light = Memory::createDeviceObject<Light>();
            Memory::copyHost2DeviceObject<Light>(light, dev_light);
            dev_lights.push_back(dev_light);
            delete light;
        }

        Memory::copyHost2DeviceArray(host_lights.size(), dev_lights.data(), scene.lights);

        return scene;
    }

    void
    SceneLoader::destroyScene(Scene& scene)
    {
        Geometry** host_scene = Memory::createHostArray<Geometry*>(scene.scene_size);
        Light** host_lights = Memory::createHostArray<Light*>(scene.light_count);
        Memory::copyDevice2HostArray(scene.scene_size, scene.geometry, host_scene);
        Memory::copyDevice2HostArray(scene.light_count, scene.lights, host_lights);

        for (uint32_t i = 0; i < scene.scene_size; ++i)
        {
            Geometry geom;
            Memory::copyDevice2HostObject(host_scene[i], &geom);
            switch (geom.type)
            {
                case GeometryType::PLANE:
                {
                    Memory::destroyDeviceObject<Plane>(static_cast<Plane*>(host_scene[i]));
                }
                break;
                case GeometryType::SPHERE:
                {
                    Memory::destroyDeviceObject<Sphere>(static_cast<Sphere*>(host_scene[i]));
                }
                break;
                case GeometryType::QUAD:
                {
                    Memory::destroyDeviceObject<Quad>(static_cast<Quad*>(host_scene[i]));
                }
                break;
                case GeometryType::TRIANGLE:
                {
                    Memory::destroyDeviceObject<Triangle>(static_cast<Triangle*>(host_scene[i]));
                }
                break;
                case GeometryType::MESH:
                {
                    Mesh mesh;
                    Memory::copyDevice2HostObject<Mesh>(static_cast<Mesh*>(host_scene[i]), &mesh);
                    Memory::destroyDeviceObject<Mesh>(static_cast<Mesh*>(host_scene[i]));

                    Memory::destroyDeviceArray<Triangle>(mesh.triangles());
                }
                break;
            }
        }

        for (uint32_t i = 0; i < scene.light_count; ++i)
        {
            Memory::destroyDeviceObject<Light>(host_lights[i]);
        }

        if (scene.useEnvironmentMap)
        {
            Image<Vector3float>::destroyDeviceObject(scene.environment);
        }

        Memory::destroyDeviceArray<Geometry*>(scene.geometry);
        Memory::destroyDeviceArray<Light*>(scene.lights);
        Memory::destroyHostArray<Geometry*>(host_scene);
        Memory::destroyHostArray<Light*>(host_lights);
    }

} //namespace cupbr