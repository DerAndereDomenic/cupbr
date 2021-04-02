#include <iostream>
#include <DataStructure/RenderBuffer.cuh>
#include <GL/GLRenderer.cuh>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <Core/KernelHelper.cuh>

__global__ void fillBuffer(RenderBuffer img, uint8_t time)
{
    uint32_t tid = ThreadHelper::globalThreadIndex();

    if(tid >= img.size())
    {
        return;
    }

    img[tid] = Vector4uint32_t(time,0,time,255);
}

int main()
{
    cudaSafeCall(cudaSetDevice(0));

    RenderBuffer img = RenderBuffer::createDeviceObject(640, 480);
    KernelSizeHelper::KernelSize config = KernelSizeHelper::configure(img.size());

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

    uint8_t time = 0;

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        /* Render here */
        glClear(GL_COLOR_BUFFER_BIT);

        ++time;
        fillBuffer<<<config.blocks, config.threads>>>(img,time);
        cudaSafeCall(cudaDeviceSynchronize());
        renderer.renderTexture(img);

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();
    }

    glfwTerminate();

    RenderBuffer::destroyDeviceObject(img);

    Memory::allocator()->printStatistics();

    return 0;
}