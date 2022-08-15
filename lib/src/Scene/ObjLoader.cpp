#include <Scene/ObjLoader.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <algorithm>
#include <Core/Memory.h>
#include <Geometry/Triangle.h>

namespace cupbr
{
    Mesh*
    ObjLoader::loadObj(const char* path, const Vector3float& position, const Vector3float& scale)
    {
        std::ifstream file;
        file.open(path);

        std::string line;

        std::vector<Vector3float> vertices;
        std::vector<Vector3float> normals;
        std::vector<Vector3uint32_t> indices_vertex;
        std::vector<Vector3uint32_t> indices_normals;

        Vector3float minimum = INFINITY;
        Vector3float maximum = -INFINITY;

        while (std::getline(file, line))
        {
            if (line[0] == 'v' && line[1] == ' ')
            {
                std::stringstream ss(line.c_str() + 2);
                std::string item;
                std::vector<float> result;

                while (std::getline(ss, item, ' '))
                {
                    result.push_back(std::stof(item));
                }
                Vector3float vertex = Vector3float(scale.x * result[0], scale.y * result[1], scale.z * result[2]) + position;
                vertices.push_back(vertex);
                maximum.x = vertex.x > maximum.x ? vertex.x : maximum.x;
                maximum.y = vertex.y > maximum.y ? vertex.y : maximum.y;
                maximum.z = vertex.z > maximum.z ? vertex.z : maximum.z;

                minimum.x = vertex.x < minimum.x ? vertex.x : minimum.x;
                minimum.y = vertex.y < minimum.y ? vertex.y : minimum.y;
                minimum.z = vertex.z < minimum.z ? vertex.z : minimum.z;
            }
            else if(line[0] == 'v' && line[1] == 'n' && line[2] == ' ')
            {
                std::stringstream ss(line.c_str() + 3);
                std::string item;
                std::vector<float> result;

                while (std::getline(ss, item, ' '))
                {
                    result.push_back(std::stof(item));
                }
                Vector3float normal = Math::normalize(Vector3float(result[0], result[1], result[2]));
                normals.push_back(normal);
            }
            else if (line[0] == 'f' && line[1] == ' ')
            {
                std::string s = std::string(line.c_str() + 2);
                std::replace(s.begin(), s.end(), ' ', '/');

                std::stringstream ss(s);
                std::string item;
                std::vector<uint32_t> result;

                while (std::getline(ss, item, '/'))
                {
                    result.push_back(std::stoi(item));
                }

                indices_vertex.push_back(Vector3uint32_t(result[0] - 1, result[3] - 1, result[6] - 1));
                indices_normals.push_back(Vector3uint32_t(result[2] - 1, result[5] - 1, result[8] - 1));
            }
        }

        file.close();

        Triangle* host_triangles = Memory::createHostArray<Triangle>(static_cast<uint32_t>(indices_vertex.size()));
        Triangle* dev_triangles = Memory::createDeviceArray<Triangle>(static_cast<uint32_t>(indices_vertex.size()));

        for (uint32_t i = 0; i < indices_vertex.size(); ++i)
        {
            Vector3uint32_t index = indices_vertex[i];
            Vector3uint32_t normal_index = indices_normals[i];
            host_triangles[i] = Triangle(vertices[index.x], 
                                         vertices[index.y], 
                                         vertices[index.z],
                                         normals[normal_index.x],
                                         normals[normal_index.y],
                                         normals[normal_index.z]);
        }

        Memory::copyHost2DeviceArray<Triangle>(static_cast<uint32_t>(indices_vertex.size()), host_triangles, dev_triangles);

        Memory::destroyHostArray<Triangle>(host_triangles);

        Mesh host_mesh(dev_triangles, static_cast<uint32_t>(indices_vertex.size()), minimum, maximum);
        Mesh* mesh = Memory::createHostObject<Mesh>();
        Memory::copyHost2HostObject<Mesh>(&host_mesh, mesh);

        return mesh;
    }

} //namespace cupbr