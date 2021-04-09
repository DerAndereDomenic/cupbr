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
    Vector3float intersection_point = Vector3float(intersection);

    //Lighting
    const Vector3float lightPos(1,2,-2);

    Vector3float brdf = Vector3float(1,1,1)/static_cast<float>(M_PI); //Albedo/pi
    Vector3float lightIntensity = Vector3float(10,10,10); //White light
    Vector3float lightDir = Math::normalize(lightPos - intersection_point);
    float d = Math::norm(intersection_point-lightPos);
    Vector3float lightRadiance = lightIntensity/(d*d);
    float cosTerm = max(0.0f,Math::dot(sphere.getNormal(intersection_point), lightDir));
    Vector3float radiance = brdf*lightRadiance*cosTerm;

    //Tone mapping

    Vector3uint8_t color(0);

    if(intersection.w != INFINITY)
    {
        float mapped_red = powf(1.0 - expf(-radiance.x), 1.0f/2.2f);
        float mapped_green = powf(1.0 - expf(-radiance.y), 1.0f/2.2f);
        float mapped_blue = powf(1.0 - expf(-radiance.z), 1.0f/2.2);

        uint8_t red = mapped_red > 1.0 ? 255 : static_cast<uint8_t>(mapped_red*255.0f);
        uint8_t green = mapped_green > 1.0 ? 255 : static_cast<uint8_t>(mapped_green*255.0f);
        uint8_t blue = mapped_blue > 1.0 ? 255 : static_cast<uint8_t>(mapped_blue*255.0f);

        color = Vector3uint8_t(red, green, blue);
    }

    img[tid] = Vector4uint8_t(color,255);
}

int main()
{
    const uint32_t width = 1024, height = 1024;

    cudaSafeCall(cudaSetDevice(0));

    RenderBuffer img = RenderBuffer::createDeviceObject(width, height);
    KernelSizeHelper::KernelSize config = KernelSizeHelper::configure(img.size());

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

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    GLRenderer renderer = GLRenderer::createHostObject(width, height);
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

        camera.processInput(window);

        if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        {
            glfwDestroyWindow(window);
            break;
        }
    }

    glfwTerminate();

    RenderBuffer::destroyDeviceObject(img);

    Memory::allocator()->printStatistics();

    return 0;
}