#include <iostream>
#include <GL/GLRenderer.cuh>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <Core/KernelHelper.cuh>
#include <DataStructure/Camera.cuh>

#include <Geometry/Sphere.cuh>

__global__ void fillBuffer(RenderBuffer img, const Camera camera)
{
    uint32_t tid = ThreadHelper::globalThreadIndex();

    if(tid >= img.size())
    {
        return;
    }

    const float width = img.width();
    const float height = img.height();
    const Vector2uint32_t pixel = ThreadHelper::index2pixel(tid, width, height);

    const float ratio_x = 2.0f*(static_cast<float>(pixel.x)/width - 0.5f);
    const float ratio_y = 2.0f*(static_cast<float>(pixel.y)/height - 0.5f);

    const Vector3float world_pos = camera.position() + camera.zAxis() + ratio_x*camera.xAxis() + ratio_y*camera.yAxis();

    Sphere sphere(Vector3float(0,0,2), 1);
    Ray ray(camera.position(), world_pos - camera.position());
    
    //Compute intersection
    Vector4float intersection = sphere.computeRayIntersection(ray);

    //"Tone mapping"

    int8_t ratio = intersection.w == INFINITY ? 0 : 255;

    img[tid] = Vector4uint32_t(ratio, ratio, ratio,255);
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
    Camera camera;

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        /* Render here */
        glClear(GL_COLOR_BUFFER_BIT);

        fillBuffer<<<config.blocks, config.threads>>>(img,camera);
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