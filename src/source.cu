#include <iostream>

#include <GL/GLRenderer.cuh>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <Core/KernelHelper.cuh>
#include <Core/Tracing.cuh>

#include <DataStructure/Camera.cuh>

#include <Geometry/Sphere.cuh>
#include <Geometry/Plane.cuh>

#include <Scene/SceneLoader.cuh>

#include <Renderer/ToneMapper.cuh>

__global__ void fillBuffer(Image<Vector3float> img, const Scene scene, const uint32_t scene_size, const Camera camera)
{
    uint32_t tid = ThreadHelper::globalThreadIndex();

    if(tid >= img.size())
    {
        return;
    }

    Ray ray = Tracing::launchRay(tid, img.width(), img.height(), camera);

    //Scene
    const Vector3float lightPos(0.0f,0.9f,2.0f);

    Vector4float intersection(INFINITY);
    Vector3float normal;
    Material material;
    for(uint32_t i = 0; i < scene_size; ++i)
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
                    material = plane->material;
                    intersection = intersection_plane;
                    normal = plane->getNormal(Vector3float(intersection));
                }
            }
            break;
            case GeometryType::SPHERE:
            {
                Sphere *sphere = static_cast<Sphere*>(scene_element);
                Vector4float intersection_sphere = sphere->computeRayIntersection(ray);
                if(intersection_sphere.w < intersection.w)
                {
                    material = sphere->material;
                    intersection = intersection_sphere;
                    normal = sphere->getNormal(Vector3float(intersection));
                }
            }
            break;
        }
    }

    Vector3float intersection_point = Vector3float(intersection);

    Vector3float inc_dir = Math::normalize(camera.position() - intersection_point);
    Vector3float lightDir = Math::normalize(lightPos - intersection_point);


    //Lighting

    Vector3float brdf = material.brdf(intersection_point, inc_dir, lightDir, normal);
    Vector3float lightIntensity = Vector3float(10,10,10); //White light
    float d = Math::norm(intersection_point-lightPos);
    Vector3float lightRadiance = lightIntensity/(d*d);
    float cosTerm = max(0.0f,Math::dot(normal, lightDir));
    Vector3float radiance = brdf*lightRadiance*cosTerm;

    //Shadow
    if(intersection.w != INFINITY)
    {
        Ray shadow_ray(intersection_point-EPSILON*ray.direction(), lightDir);
        
        for(uint32_t i = 0; i < scene_size; ++i)
        {
            Geometry* scene_element = scene[i];
            switch(scene_element->type)
            {
                case GeometryType::PLANE:
                {
                    Plane *plane = static_cast<Plane*>(scene[i]);
                    Vector4float intersection_plane = plane->computeRayIntersection(shadow_ray);
                    if(intersection_plane.w != INFINITY && intersection_plane.w < d)
                    {
                        radiance = Vector3float(0);
                        break;
                    }
                }
                break;
                case GeometryType::SPHERE:
                {
                    Sphere *sphere = static_cast<Sphere*>(scene_element);
                    Vector4float intersection_sphere = sphere->computeRayIntersection(shadow_ray);
                    if(intersection_sphere.w != INFINITY && intersection_sphere.w < d)
                    {
                        radiance = Vector3float(0);
                        break;
                    }
                }
                break;
            }
        }
    }

    img[tid] = radiance;
}

int main()
{
    bool edit = true;
    bool pressed = false;
    const uint32_t width = 1024, height = 1024;

    cudaSafeCall(cudaSetDevice(0));

    Image<Vector3float> img = Image<Vector3float>::createDeviceObject(width, height);
    KernelSizeHelper::KernelSize config = KernelSizeHelper::configure(img.size());
    ToneMapper reinhard_mapper;
    reinhard_mapper.registerImage(&img);

    GLFWwindow* window;

    /* Initialize the library */
    if (!glfwInit())
        return -1;

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(width, height, "Hello World", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    if (glewInit() != GLEW_OK)
	{
		std::cout <<"RENDERER::GLEWINIT::ERROR\n";
	}

    GLRenderer renderer(width, height);
    Camera camera;

    uint32_t scene_size;
    Scene scene = SceneLoader::cornellBoxSphere(&scene_size);

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        /* Render here */
        glClear(GL_COLOR_BUFFER_BIT);

        fillBuffer<<<config.blocks, config.threads>>>(img,scene,scene_size,camera);
        cudaSafeCall(cudaDeviceSynchronize());

        reinhard_mapper.toneMap();
        renderer.renderTexture(reinhard_mapper.getRenderBuffer());

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();

        if(!edit)
            camera.processInput(window);

        if(glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS && !pressed)
        {
            pressed = true;
            edit = !edit;
            if(edit)
            {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            }
            else
            {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            }
        }

        if(glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_RELEASE)
        {
            pressed = false;
        }

        if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        {
            glfwDestroyWindow(window);
            break;
        }
    }

    glfwTerminate();

    Image<Vector3float>::destroyDeviceObject(img);
    SceneLoader::destroyCornellBoxSphere(scene);

    //TODO
    reinhard_mapper.~ToneMapper();

    Memory::allocator()->printStatistics();

    return 0;
}