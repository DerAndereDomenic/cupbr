#include <Scene/SceneLoader.h>
#include <Core/Memory.h>
#include <Geometry/Plane.h>
#include <Geometry/Sphere.h>
#include <Geometry/Quad.h>
#include <Geometry/Triangle.h>
#include <Geometry/Mesh.h>
#include <Scene/ObjLoader.h>
#include <Scene/SDF.h>
#include <tinyxml2.h>
#include <vector>
#include <sstream>
#include <filesystem>

#include <Core/Plugin.h>

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
                                                                                                                        
        Material*
        loadMaterial(const tinyxml2::XMLElement* material_ptr, Scene* scene)
        {
            const char* type = material_ptr->FirstChildElement("name")->GetText();

            //Load material properties

            Properties properties;
            properties.setProperty("name", std::string(type));

            auto current_element = material_ptr->FirstChild();
            while(current_element != nullptr)
            {
                if(strcmp(current_element->Value(), "name") != 0)
                {
                    auto current_node = material_ptr->FirstChildElement(current_element->Value());

                    auto attribute = current_node->FirstAttribute()->Value();
                    if(strcmp(attribute, "vec3") == 0)
                    {
                        properties.setProperty(current_element->Value(), string2vector(current_node->GetText()));
                    }
                    else if(strcmp(attribute, "float") == 0)
                    {
                        properties.setProperty(current_element->Value(), std::stof(current_node->GetText()));
                    }
                    else if(strcmp(attribute, "int") == 0)
                    {
                        properties.setProperty(current_element->Value(), std::stoi(current_node->GetText()));
                    }
                    else if(strcmp(attribute, "string") == 0)
                    {
                        properties.setProperty(current_element->Value(), std::string(current_node->GetText()));
                    }
                    else if(strcmp(attribute, "bool") == 0)
                    {
                        properties.setProperty(current_element->Value(), static_cast<bool>(std::stoi(current_node->GetText())));
                    }
                    else
                    {
                        //TODO:
                        std::cerr << "Datatype not implemented: " << attribute << std::endl;
                        std::cerr << "Supported types: string, vec3, float, int, bool" << std::endl;
                        std::cerr << "If you think your datatype should be supported as well open a pull request or github issue!" << std::endl;
                        return nullptr;
                    }
                }

                current_element = current_element->NextSibling();
            }


            std::shared_ptr<PluginInstance> instance = PluginManager::getPlugin(std::string(type));
            Material* material = reinterpret_cast<Material*>(instance->createDeviceObject(&properties));
            scene->properties.push_back(properties);
            return material;
        }

        Scene* load_geometry_scene(const tinyxml2::XMLDocument& doc)
        {
            GeometryScene* scene = Memory::createHostObject<GeometryScene>();
            std::vector<Geometry*> host_geometry;
            std::vector<Light*> host_lights;

            //Load geometry
            const tinyxml2::XMLElement* current_geometry = doc.FirstChildElement("geometry");

            uint32_t id = 0;
            while (current_geometry != nullptr)
            {
                if (strcmp(current_geometry->Name(), "geometry") != 0)
                {
                    current_geometry = current_geometry->NextSiblingElement();
                    continue;
                }
                const char* type = current_geometry->FirstChildElement("type")->GetText();
                if (strcmp(type, "PLANE") == 0)
                {
                    const char* position_string = current_geometry->FirstChildElement("position")->GetText();
                    const char* normal_string = current_geometry->FirstChildElement("normal")->GetText();
                    Vector3float position = detail::string2vector(position_string);
                    Vector3float normal = detail::string2vector(normal_string);
                    Material* mat = detail::loadMaterial(current_geometry->FirstChildElement("material"), scene);

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
                    Material* mat = detail::loadMaterial(current_geometry->FirstChildElement("material"), scene);

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

                    Material* mat = detail::loadMaterial(current_geometry->FirstChildElement("material"), scene);

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

                    Material* mat = detail::loadMaterial(current_geometry->FirstChildElement("material"), scene);

                    Triangle* geom = new Triangle(vertex1, vertex2, vertex3);
                    geom->material = mat;

                    host_geometry.push_back(geom);
                }
                else if (strcmp(type, "MESH") == 0)
                {
                    const char* path_string = current_geometry->FirstChildElement("path")->GetText();
                    Vector3float position = 0;
                    Vector3float scale = 1;
                    const tinyxml2::XMLElement* position_string = current_geometry->FirstChildElement("position");
                    const tinyxml2::XMLElement* scale_string = current_geometry->FirstChildElement("scale");

                    if (position_string != NULL)
                    {
                        position = detail::string2vector(position_string->GetText());
                    }

                    if (scale_string != NULL)
                    {
                        scale = detail::string2vector(scale_string->GetText());
                    }

                    Mesh* geom = ObjLoader::loadObj(path_string, position, scale);

                    Material* mat = detail::loadMaterial(current_geometry->FirstChildElement("material"), scene);

                    geom->material = mat;

                    host_geometry.push_back(geom);
                }
                else
                {
                    std::cerr << "Error while loading scene: " << type << " is not a valid geometry type!" << std::endl;
                    return nullptr;
                }
                host_geometry.back()->setID(id++);
                current_geometry = current_geometry->NextSiblingElement();
            }
            scene->scene_size = host_geometry.size();
            scene->geometry = Memory::createDeviceArray<Geometry*>(scene->scene_size);

            const tinyxml2::XMLElement* current_light = doc.FirstChildElement("light");
            while (current_light != nullptr)
            {
                if (strcmp(current_light->Name(), "light") != 0)
                {
                    current_light = current_light->NextSiblingElement();
                    continue;
                }
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
                    return nullptr;
                }

                const tinyxml2::XMLElement* position_string = current_light->FirstChildElement("position");
                const tinyxml2::XMLElement* intensity_string = current_light->FirstChildElement("intensity");
                const tinyxml2::XMLElement* radiance_string = current_light->FirstChildElement("radiance");
                const tinyxml2::XMLElement* extend1_string = current_light->FirstChildElement("extend1");
                const tinyxml2::XMLElement* extend2_string = current_light->FirstChildElement("extend2");

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

                current_light = current_light->NextSiblingElement();
            }
            scene->light_count = host_lights.size();
            scene->lights = Memory::createDeviceArray<Light*>(scene->light_count);

            const tinyxml2::XMLElement* volume_head = doc.FirstChildElement("volume");
            if (volume_head != NULL)
            {
                const char* sigma_s_string = volume_head->FirstChildElement("sigma_s")->GetText();
                const char* sigma_a_string = volume_head->FirstChildElement("sigma_a")->GetText();
                const char* g_string = volume_head->FirstChildElement("g")->GetText();

                Volume vol;

                vol.sigma_s = detail::string2vector(sigma_s_string);
                vol.sigma_a = detail::string2vector(sigma_a_string);
                vol.g = std::stof(g_string);

                scene->volume = vol;
            }

            const tinyxml2::XMLElement* environment_head = doc.FirstChildElement("environment");
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

                scene->useEnvironmentMap = true;
                scene->environment = dbuffer;

                stbi_image_free(data);

                Image<Vector3float>::destroyHostObject(buffer);
            }

            //Transfer data to device
            //TODO: Backend
            std::vector<Geometry*> dev_geometry;
            for (uint32_t i = 0; i < host_geometry.size(); ++i)
            {
                Geometry* geom = host_geometry[i];
                switch (geom->type)
                {
                case GeometryType::SPHERE:
                {
                    Sphere* dev_geom = Memory::createDeviceObject<Sphere>();
                    Memory::copyHost2DeviceObject<Sphere>(static_cast<Sphere*>(geom), dev_geom);
                    dev_geometry.push_back(dev_geom);
                }
                break;
                case GeometryType::PLANE:
                {
                    Plane* dev_geom = Memory::createDeviceObject<Plane>();
                    Memory::copyHost2DeviceObject<Plane>(static_cast<Plane*>(geom), dev_geom);
                    dev_geometry.push_back(dev_geom);
                }
                break;
                case GeometryType::QUAD:
                {
                    Quad* dev_geom = Memory::createDeviceObject<Quad>();
                    Memory::copyHost2DeviceObject<Quad>(static_cast<Quad*>(geom), dev_geom);
                    dev_geometry.push_back(dev_geom);
                }
                break;
                case GeometryType::TRIANGLE:
                {
                    Triangle* dev_geom = Memory::createDeviceObject<Triangle>();
                    Memory::copyHost2DeviceObject<Triangle>(static_cast<Triangle*>(geom), dev_geom);
                    dev_geometry.push_back(dev_geom);
                }
                break;
                case GeometryType::MESH:
                {
                    Mesh* dev_geom = Memory::createDeviceObject<Mesh>();
                    Memory::copyHost2DeviceObject<Mesh>(static_cast<Mesh*>(geom), dev_geom);
                    dev_geometry.push_back(dev_geom);
                }
                break;
                }
            }

            if (host_geometry.size() > 0)
            {
                BoundingVolumeHierarchy bvh(host_geometry, dev_geometry);
                scene->bvh = Memory::createDeviceObject<BoundingVolumeHierarchy>();
                Memory::copyHost2DeviceObject<BoundingVolumeHierarchy>(&bvh, scene->bvh);

                Memory::copyHost2DeviceArray(host_geometry.size(), dev_geometry.data(), scene->geometry);
            }

            std::vector<Light*> dev_lights;
            for (uint32_t i = 0; i < host_lights.size(); ++i)
            {
                Light* light = host_lights[i];
                Light* dev_light = Memory::createDeviceObject<Light>();
                Memory::copyHost2DeviceObject<Light>(light, dev_light);
                dev_lights.push_back(dev_light);
                delete light;
            }

            //Delete temp host objects
            for (uint32_t i = 0; i < host_geometry.size(); ++i)
            {
                Geometry* geom = host_geometry[i];
                switch (geom->type)
                {
                case GeometryType::MESH:
                {
                    Memory::destroyHostObject<Mesh>(static_cast<Mesh*>(geom));
                }
                break;
                default:
                {
                    delete geom;
                }
                break;
                }
            }

            Memory::copyHost2DeviceArray(host_lights.size(), dev_lights.data(), scene->lights);

            return scene;
        }

        Scene* load_sdf_scene(const tinyxml2::XMLDocument& doc)
        {
            SDFScene* scene = Memory::createHostObject<SDFScene>();

            const tinyxml2::XMLElement* sdf_head = doc.FirstChildElement("sdf");
            while (sdf_head != nullptr)
            {
                Properties properties;
            
                auto current_element = sdf_head->FirstChild();
                while (current_element != nullptr)
                {
                    auto current_node = sdf_head->FirstChildElement(current_element->Value());
                    auto attribute = current_node->FirstAttribute()->Value();
                    if (strcmp(attribute, "vec3") == 0)
                    {
                        properties.setProperty(current_element->Value(), detail::string2vector(current_node->GetText()));
                    }
                    else if (strcmp(attribute, "float") == 0)
                    {
                        properties.setProperty(current_element->Value(), std::stof(current_node->GetText()));
                    }
                    else if (strcmp(attribute, "int") == 0)
                    {
                        properties.setProperty(current_element->Value(), std::stoi(current_node->GetText()));
                    }
                    else if (strcmp(attribute, "string") == 0)
                    {
                        properties.setProperty(current_element->Value(), std::string(current_node->GetText()));
                    }
                    else if (strcmp(attribute, "bool") == 0)
                    {
                        properties.setProperty(current_element->Value(), static_cast<bool>(std::stoi(current_node->GetText())));
                    }
                    else
                    {
                        //TODO:
                        std::cerr << "Datatype not implemented: " << attribute << std::endl;
                        std::cerr << "Supported types: string, vec3, float, int, bool" << std::endl;
                        std::cerr << "If you think your datatype should be supported as well open a pull request or github issue!" << std::endl;
                        return nullptr;
                    }
            
                    current_element = current_element->NextSibling();
                }
            
                std::shared_ptr<PluginInstance> instance = PluginManager::getPlugin(properties.getProperty(std::string("name"), std::string("")));
                SDF* sdf = reinterpret_cast<SDF*>(instance->createDeviceObject(&properties));
            
                scene->sdf = sdf;
                scene->properties.push_back(properties);
            
                sdf_head = sdf_head->NextSiblingElement();
            
            }

            return scene;
        }

        void destroy_geometry_scene(GeometryScene* scene)
        {
            Geometry** host_scene = Memory::createHostArray<Geometry*>(scene->scene_size);
            Light** host_lights = Memory::createHostArray<Light*>(scene->light_count);
            Memory::copyDevice2HostArray(scene->scene_size, scene->geometry, host_scene);
            Memory::copyDevice2HostArray(scene->light_count, scene->lights, host_lights);

            for (uint32_t i = 0; i < scene->scene_size; ++i)
            {
                Geometry geom;
                Memory::copyDevice2HostObject(host_scene[i], &geom);
                //Don't use memory API because materials are obtained by plugin
                cudaFree(geom.material);
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

            for (uint32_t i = 0; i < scene->light_count; ++i)
            {
                Memory::destroyDeviceObject<Light>(host_lights[i]);
            }

            if (scene->useEnvironmentMap)
            {
                Image<Vector3float>::destroyDeviceObject(scene->environment);
            }

            BoundingVolumeHierarchy* host_bvh = Memory::createHostObject<BoundingVolumeHierarchy>();
            Memory::copyDevice2HostObject<BoundingVolumeHierarchy>(scene->bvh, host_bvh);
            host_bvh->destroy();
            Memory::destroyDeviceObject<BoundingVolumeHierarchy>(scene->bvh);

            Memory::destroyDeviceArray<Geometry*>(scene->geometry);
            Memory::destroyDeviceArray<Light*>(scene->lights);
            Memory::destroyHostArray<Geometry*>(host_scene);
            Memory::destroyHostArray<Light*>(host_lights);
            Memory::destroyHostObject<BoundingVolumeHierarchy>(host_bvh);

            Memory::destroyHostObject<GeometryScene>(scene);
        }

        void destroy_sdf_scene(SDFScene* scene)
        {
            cudaFree(scene->sdf);
            Memory::destroyHostObject<SDFScene>(scene);
        }

    } //namespace detail

    Scene*
    SceneLoader::loadFromFile(const std::string& path)
    {
        tinyxml2::XMLDocument doc;
        tinyxml2::XMLError error;

        do
        {
            error = doc.LoadFile(path.c_str());

            if (error != tinyxml2::XML_SUCCESS && !std::filesystem::exists(path))
            {
                std::cerr << "Failed to load XML file: " << path << ". Error code: " << error << "\n";
                return nullptr;
            }
        } while (error == tinyxml2::XML_ERROR_FILE_NOT_FOUND && std::filesystem::exists(path));

        bool has_geometry = doc.FirstChildElement("geometry") != nullptr;
        bool has_sdf = doc.FirstChildElement("sdf") != nullptr;

        if (has_geometry && has_sdf)
        {
            std::cerr << "ERROR: Scene contains both geometry and sdf information!\n";
            return nullptr;
        }
        else if (has_geometry)
        {
            return detail::load_geometry_scene(doc);
        }
        else if (has_sdf)
        {
            return detail::load_sdf_scene(doc);
        }

        return nullptr;
    }

    void 
    SceneLoader::reinitializeScene(Scene* scene_)
    {
        GeometryScene* scene = dynamic_cast<GeometryScene*>(scene_);
        Geometry** host_scene = Memory::createHostArray<Geometry*>(scene->scene_size);
        Memory::copyDevice2HostArray(scene->scene_size, scene->geometry, host_scene);

        for(uint32_t i = 0; i < scene->scene_size; ++i)
        {
            Properties& properties = scene->properties[i];
            std::string name = properties.getProperty<std::string>("name").value();
            std::shared_ptr<PluginInstance> instance = PluginManager::getPlugin(name);
            //From dll -> don't use memory API

            Geometry geom;
            Memory::copyDevice2HostObject(host_scene[i], &geom);

            cudaFree(geom.material);
            geom.material = reinterpret_cast<Material*>(instance->createDeviceObject(&properties));

            Memory::copyHost2DeviceObject(&geom, host_scene[i]);
        }

        Memory::destroyHostArray<Geometry*>(host_scene);
    }

    void
    SceneLoader::destroyScene(Scene* scene_)
    {
        GeometryScene* geom_scene = dynamic_cast<GeometryScene*>(scene_);
        SDFScene* sdf_scene = dynamic_cast<SDFScene*>(scene_);

        if (geom_scene != nullptr)
        {
            detail::destroy_geometry_scene(geom_scene);
        }
        else if (sdf_scene != nullptr)
        {
            detail::destroy_sdf_scene(sdf_scene);
        }
    }

} //namespace cupbr