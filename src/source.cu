#include <iostream>

#include <CUPBR.h>

using namespace cupbr;

int run(int argc, char* argv[])
{
    const uint32_t width = 1024, height = 1024, menu_width = 400;

    cudaSafeCall(cudaSetDevice(0));

    //TODO: Find Plugin automatically
    PluginManager::loadPlugin("MaterialLambert");
    PluginManager::loadPlugin("MaterialGGX");
    PluginManager::loadPlugin("MaterialPhong");
    PluginManager::loadPlugin("MaterialMirror");
    PluginManager::loadPlugin("MaterialGlass");

    Scene scene;
    
    if(argc == 1)
    {
        scene = SceneLoader::loadFromFile("res/Scenes/CornellBoxSphereAreaLight.xml");
    }
    else if(argc == 2)
    {
        scene = SceneLoader::loadFromFile(argv[1]);
    }

    PBRenderer pbrenderer;
    pbrenderer.setOutputSize(width, height);
    pbrenderer.registerScene(&scene);

    ToneMapper mapper(ToneMappingType::REINHARD);
    mapper.registerImage(pbrenderer.getOutputImage());

    Window window("CUPBR", width + menu_width, height);

    GLRenderer renderer(width, height);
    Camera camera(width,height);
    Interactor interactor;
    interactor.registerWindow(&window, menu_width);
    interactor.registerCamera(&camera);
    interactor.registerScene(&scene);
    interactor.registerRenderer(&pbrenderer);
    interactor.registerToneMapper(&mapper);

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
        
        /* Render here */
        glClear(GL_COLOR_BUFFER_BIT);

        pbrenderer.render(&camera);

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