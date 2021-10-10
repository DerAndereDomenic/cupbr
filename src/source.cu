#include <iostream>
#include <chrono>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

#include <CUPBR.cuh>

using namespace cupbr;

int run(int argc, char* argv[])
{
    bool edit = true;
    bool pressed = false;
    const uint32_t width = 1024, height = 1024;

    cudaSafeCall(cudaSetDevice(0));

    Scene scene;
    
    if(argc == 1)
    {
        scene = SceneLoader::loadFromFile("res/Scenes/CornellBoxSphereAreaLight.xml");
    }
    else if(argc == 2)
    {
        scene = SceneLoader::loadFromFile(argv[1]);
    }

    PBRenderer pbrenderer(RenderingMethod::PATHTRACER);
    pbrenderer.setOutputSize(width, height);
    pbrenderer.registerScene(&scene);

    PostProcessor postprocessor;
    postprocessor.registerImage(pbrenderer.getOutputImage());

    Vector3float kernel_data[9] =
    {
        Vector3float(1.0),Vector3float(2.0),Vector3float(1.0),
        Vector3float(0),Vector3float(0),Vector3float(0),
        Vector3float(-1.0),Vector3float(-2.0),Vector3float(-1.0),
    };

    Image<Vector3float> kernel = Image<Vector3float>::createDeviceObject(kernel_data, 3, 3);

    ToneMapper mapper(ToneMappingType::REINHARD);
    mapper.registerImage(postprocessor.getPostProcessBuffer());

    GLFWwindow* window;

    /* Initialize the library */
    if (!glfwInit())
        return -1;

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(width, height, "CUPBR", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);

    if (!gladLoadGL()) {
        std::cout << "Failed to initialize OpenGL context" << std::endl;
        return -1;
    }

    GLRenderer renderer(width, height);
    Camera camera(width,height);
    Interactor interactor(pbrenderer.getMethod());
    interactor.registerWindow(window);
    interactor.registerCamera(camera);
    interactor.registerScene(&scene);

    float time = 0.0f;
    uint32_t frame_counter = 0;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    bool post_proc = true;

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        interactor.handleInteraction();

        if(post_proc != interactor.usePostProcessing())
        {
            if(interactor.usePostProcessing())
            {
                mapper.registerImage(postprocessor.getPostProcessBuffer());
            }
            else
            {
                mapper.registerImage(pbrenderer.getOutputImage());
            }
            post_proc = interactor.usePostProcessing();
        }
        

        mapper.setExposure(interactor.getExposure());

        if(interactor.getRenderingMethod() != pbrenderer.getMethod() || interactor.updated())
        {
            pbrenderer.setMethod(interactor.getRenderingMethod());
        }

        if(interactor.getToneMapping() != mapper.getType())
        {
            mapper.setType(interactor.getToneMapping());
        }
        
        ImGui::Render();
        /* Render here */
        glClear(GL_COLOR_BUFFER_BIT);

        pbrenderer.render(camera);

        //postprocessor.filter(kernel);
        if(interactor.usePostProcessing())
            postprocessor.bloom(interactor.getThreshold());

        mapper.toneMap();
        renderer.renderTexture(mapper.getRenderBuffer());
        
        ++frame_counter;

        if(frame_counter%30 == 0)
        {
            printf("\rRender time: %fms : %ffps", time/1000.0f, 1000000.0f/time);
            fflush(stdout);
        }

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();

        if(!edit)
            camera.processInput(window, time);

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

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        time = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count();
    }
    printf("\n");

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwTerminate();

    SceneLoader::destroyScene(scene);

    mapper.saveToFile("bin/output.bmp");

    Image<Vector3float>::destroyDeviceObject(kernel);

    printf("Rendered Frames: %i\n", frame_counter);
    return 0;
}

int main(int argc, char* argv[])
{
    int exit =  run(argc, argv);
    Memory::printStatistics();
    return exit;
}