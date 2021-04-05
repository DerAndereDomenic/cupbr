#include <iostream>
#include <GL/GLRenderer.cuh>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <Core/KernelHelper.cuh>
#include <DataStructure/Camera.cuh>

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

    const Vector3float sphere_pos = Vector3float(0,0,2);
    const float r = 1.0f;

    const Vector3float origin = camera.position();
    Vector3float direction = world_pos - origin;
    Math::normalize(direction);
    
    //Compute intersection
    Vector3float intersection = Vector3float(INFINITY, INFINITY, INFINITY);

    float a = Math::dot(direction, direction);
    Vector3float OS = origin - sphere_pos;
    float b = 2.0f * Math::dot(direction, OS);
    float c = Math::dot(OS, OS) - r*r;
    float disc = b*b - 4*a*c;

    if(disc > 0)
    {
        float distSqrt = sqrtf(disc);
        float q = b<0?(-b-distSqrt)/2.0f : (-b+distSqrt)/2.0f;
        float t0 = q/a;
        float t1 = c/q;
        t0 = min(t0,t1);
        t1 = max(t0,t1);
        if(t1 >= 0)
        {
            float t = t0 < 0 ? t1 : t0;
            intersection = origin+t*direction;
        }
    }

    //"Tone mapping"

    int8_t ratio = intersection.x == INFINITY ? 0 : 255;

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