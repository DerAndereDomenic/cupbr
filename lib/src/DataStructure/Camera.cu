#include <DataStructure/Camera.cuh>

void
Camera::processInput(GLFWwindow* window)
{
    if(glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    {
        _position.x += 0.1f;
    }
    if(glfwGetKey(window,GLFW_KEY_A) == GLFW_PRESS)
    {
        _position.x -= 0.1f;
    }
    if(glfwGetKey(window,GLFW_KEY_S) == GLFW_PRESS)
    {
        _position.z -= 0.1f;
    }
    if(glfwGetKey(window,GLFW_KEY_W) == GLFW_PRESS)
    {
        _position.z += 0.1f;
    }
}