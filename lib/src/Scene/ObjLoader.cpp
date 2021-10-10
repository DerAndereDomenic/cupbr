#include <Scene/ObjLoader.cuh>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <algorithm>
#include <Core/Memory.cuh>
#include <Geometry/Triangle.cuh>

namespace cupbr
{
    Mesh*
        ObjLoader::loadObj(const char* path, const Vector3float& position)
    {
        std::ifstream file;
        file.open(path);

        std::string line;

        std::vector<Vector3float> vertices;
        std::vector<Vector3uint32_t> indices;

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

                vertices.push_back(Vector3float(result[0], result[1], result[2]) + position);
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

                indices.push_back(Vector3uint32_t(result[0] - 1, result[3] - 1, result[6] - 1));
            }
        }

        file.close();

        Triangle** host_triangles = Memory::allocator()->createHostArray<Triangle*>(indices.size());
        Triangle** dev_triangles = Memory::allocator()->createDeviceArray<Triangle*>(indices.size());

        for (uint32_t i = 0; i < indices.size(); ++i)
        {
            Vector3uint32_t index = indices[i];
            Triangle host_triangle(vertices[index.x], vertices[index.y], vertices[index.z]);
            Triangle* triangle = Memory::allocator()->createDeviceObject<Triangle>();
            Memory::allocator()->copyHost2DeviceObject<Triangle>(&host_triangle, triangle);
            host_triangles[i] = triangle;
        }

        Memory::allocator()->copyHost2DeviceArray<Triangle*>(indices.size(), host_triangles, dev_triangles);

        Memory::allocator()->destroyHostArray<Triangle*>(host_triangles);

        Mesh host_mesh(dev_triangles, indices.size());
        Mesh* mesh = Memory::allocator()->createHostObject<Mesh>();
        Memory::allocator()->copyHost2HostObject<Mesh>(&host_mesh, mesh);

        return mesh;
    }

} //namespace cupbr