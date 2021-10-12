#include <iostream>

#include <CUPBR.cuh>

using namespace cupbr;

int run(int argc, char* argv[])
{
    const uint32_t width = 1024, height = 1024, menu_width = 400;

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

    ToneMapper mapper(ToneMappingType::REINHARD);
    mapper.registerImage(postprocessor.getPostProcessBuffer());

    Window window("CUPBR", width + menu_width, height);

    GLRenderer renderer(width, height);
    Camera camera(width,height);
    Interactor interactor(pbrenderer.getMethod());
    interactor.registerWindow(&window, menu_width);
    interactor.registerCamera(&camera);
    interactor.registerScene(&scene);

    window.setEventCallback(std::bind(&Interactor::onEvent, &interactor, std::placeholders::_1));

    uint32_t frame_counter = 0;

    bool post_proc = true;

    glViewport(0,0, width, height);

    std::string scene_path;

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose((GLFWwindow*)window.getInternalWindow()))
    {
        window.imguiBegin();

        interactor.handleInteraction();

        if(interactor.resetScene(scene_path))
        {
            SceneLoader::destroyScene(scene);
            scene = SceneLoader::loadFromFile(scene_path);
            camera = Camera(width, height);
            pbrenderer.reset();
        }

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
        
        /* Render here */
        glClear(GL_COLOR_BUFFER_BIT);

        pbrenderer.render(&camera);

        if(interactor.usePostProcessing())
            postprocessor.bloom(interactor.getThreshold());

        mapper.toneMap();
        renderer.renderTexture(mapper.getRenderBuffer());
        
        ++frame_counter;

        if(frame_counter%30 == 0)
        {
            printf("\rRender time: %fms : %ffps", window.delta_time()*1000.0f, 1.0f/window.delta_time());
            fflush(stdout);
        }

        window.imguiEnd();

        window.spinOnce();

        if(interactor.shouldClose())
        {
            glfwDestroyWindow((GLFWwindow*)window.getInternalWindow());
            break;
        }
    }
    printf("\n");

    SceneLoader::destroyScene(scene);

    mapper.saveToFile("bin/output.bmp");

    printf("Rendered Frames: %i\n", frame_counter);
    return 0;
}

int main(int argc, char* argv[])
{
    int exit =  run(argc, argv);
    Memory::printStatistics();
    return exit;
}