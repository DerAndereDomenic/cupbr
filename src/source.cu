#include <iostream>
#include <filesystem>

#include <Core/EntryPoint.h>

using namespace cupbr;

int run(int argc, char* argv[])
{
    const uint32_t width = 1100, height = 800;

    Scene scene;
    
    std::string scene_path;
    if(argc == 1)
    {
        scene_path = "res/Scenes/CornellBoxSphereAreaLight.xml";
    }
    else if(argc == 2)
    {
        scene_path = argv[1];
    }
    scene = SceneLoader::loadFromFile(scene_path);

    PBRenderer pbrenderer;
    pbrenderer.setOutputSize(width, height);
    pbrenderer.registerScene(&scene);

    ToneMapper mapper;
    mapper.registerImage(pbrenderer.getOutputImage());

    Window window("CUPBR", width, height);

    Camera camera(static_cast<float>(width) / static_cast<float>(height));
    Interactor interactor;
    interactor.registerWindow(&window);
    interactor.registerCamera(&camera);
    interactor.registerScene(&scene);
    interactor.registerRenderer(&pbrenderer);
    interactor.registerToneMapper(&mapper);

    window.setEventCallback(std::bind(&Interactor::onEvent, &interactor, std::placeholders::_1));

    uint32_t frame_counter = 0;

    window.setViewport(0, 0, width, height);

    FileWatcher scene_path_watcher(scene_path);

    /* Loop until the user closes the window */
    while (!window.shouldClose())
    {
        window.imguiBegin();

        interactor.handleInteraction();

        if(interactor.resetScene(scene_path) || scene_path_watcher.fileUpdated())
        {
            SceneLoader::destroyScene(scene);
            scene = SceneLoader::loadFromFile(scene_path);
            scene_path_watcher.setPath(scene_path);
            pbrenderer.reset();
        } 
        
        /* Render here */
        window.clear();

        pbrenderer.render(&camera);

        mapper.toneMap();
        window.displayImage(mapper.getRenderBuffer());
        
        ++frame_counter;

        window.imguiEnd();

        window.spinOnce();

        if(interactor.shouldClose())
        {
            window.close();
            break;
        }
    }

    SceneLoader::destroyScene(scene);

    mapper.saveToFile("bin/output.bmp");

    printf("Rendered Frames: %i\n", frame_counter);
    return 0;
}