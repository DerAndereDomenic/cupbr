#include <iostream>
#include <DataStructure/RenderBuffer.cuh>
#include <GL/GLRenderer.cuh>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

int main()
{
    RenderBuffer img = RenderBuffer::createHostObject(640, 480);

    for(uint32_t i = 0; i < 640; ++i)
    {
        for(uint32_t j = 0; j < 480; ++j)
        {
            Vector2uint32_t pixel(i,j);
            img(pixel) = Vector4uint8_t(255,0,255,255);
        }
    }

    GLFWwindow* window;

    /* Initialize the library */
    if (!glfwInit())
        return -1;

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
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


    GLRenderer renderer = GLRenderer::createHostObject(640, 480);

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        /* Render here */
        glClear(GL_COLOR_BUFFER_BIT);

        renderer.renderTexture(img);

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();
    }

    glfwTerminate();

    RenderBuffer::destroyHostObject(img);

    Memory::allocator()->printStatistics();

    return 0;
}