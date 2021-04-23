#include <iostream>
#include <chrono>

#include <GL/GLRenderer.cuh>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <Core/KernelHelper.cuh>
#include <Core/Tracing.cuh>

#include <DataStructure/Camera.cuh>

#include <Scene/SceneLoader.cuh>

#include <Renderer/ToneMapper.cuh>
#include <Renderer/PBRenderer.cuh>

int main()
{
    bool edit = true;
    bool pressed = false;
    const uint32_t width = 1024, height = 1024;

    cudaSafeCall(cudaSetDevice(0));

    Scene scene = SceneLoader::cornellBoxSphereMultiLight();

    PBRenderer pbrenderer(PATHTRACER);
    pbrenderer.setOutputSize(width, height);
    pbrenderer.registerScene(scene);

    KernelSizeHelper::KernelSize config = KernelSizeHelper::configure(pbrenderer.getOutputImage()->size());
    ToneMapper reinhard_mapper(REINHARD);
    ToneMapper gamma_mapper(GAMMA);
    reinhard_mapper.registerImage(pbrenderer.getOutputImage());
    gamma_mapper.registerImage(pbrenderer.getOutputImage());

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
    float time = 0.0f;
    uint32_t frame_counter = 0;

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        /* Render here */
        glClear(GL_COLOR_BUFFER_BIT);

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        pbrenderer.render(camera);

        mapper->toneMap();
        renderer.renderTexture(mapper->getRenderBuffer());
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        time = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count();
        ++frame_counter;

        if(frame_counter%30 == 0)
        {
            printf("\rRender time: %fms : %ffps", time/1000.0f, 1000000.0f/time);
            fflush(stdout);
        }

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

        if(glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS && !pressed)
        {
            pbrenderer.setMethod(RAYTRACER);
        }

        if(glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS && !pressed)
        {
            pbrenderer.setMethod(WHITTED);
        }

        if(glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS && !pressed)
        {
            pbrenderer.setMethod(PATHTRACER);
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
    printf("\n");

    glfwTerminate();

    SceneLoader::destroyCornellBoxSphere(scene);

    //TODO
    reinhard_mapper.~ToneMapper();
    gamma_mapper.~ToneMapper();
    pbrenderer.~PBRenderer();

    Memory::allocator()->printStatistics();

    return 0;
}