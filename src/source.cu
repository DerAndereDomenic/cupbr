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

    LocalGeometry geom = Tracing::traceRay(scene, scene_size, ray);

    Vector3float inc_dir = Math::normalize(camera.position() - geom.P);
    Vector3float lightDir = Math::normalize(lightPos - geom.P);


    //Lighting

    Vector3float brdf = geom.material.brdf(geom.P, inc_dir, lightDir, geom.N);
    Vector3float lightIntensity = Vector3float(10,10,10); //White light
    float d = Math::norm(geom.P-lightPos);
    Vector3float lightRadiance = lightIntensity/(d*d);
    float cosTerm = max(0.0f,Math::dot(geom.N, lightDir));
    Vector3float radiance = brdf*lightRadiance*cosTerm;

    //Shadow
    if(geom.depth != INFINITY)
    {
        Ray shadow_ray(geom.P-EPSILON*ray.direction(), lightDir);
        
        if(!Tracing::traceVisibility(scene, scene_size, d, shadow_ray))
        {
            radiance = 0;
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
    ToneMapper reinhard_mapper(REINHARD);
    ToneMapper gamma_mapper(GAMMA);
    reinhard_mapper.registerImage(&img);
    gamma_mapper.registerImage(&img);

    ToneMapper* mapper = &reinhard_mapper;

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

        mapper->toneMap();
        renderer.renderTexture(mapper->getRenderBuffer());

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();

        if(!edit)
            camera.processInput(window);

        if(glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
        {
            mapper = &reinhard_mapper;
        }
        if(glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS)
        {
            mapper = &gamma_mapper;
        }

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
    gamma_mapper.~ToneMapper();

    Memory::allocator()->printStatistics();

    return 0;
}