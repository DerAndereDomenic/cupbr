#include <DataStructure/Camera.cuh>

void
Camera::processInput(GLFWwindow* window)
{
    if(glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    {
        _position += 0.1f*_xAxis;
    }
    if(glfwGetKey(window,GLFW_KEY_A) == GLFW_PRESS)
    {
        _position -= 0.1f*_xAxis;
    }
    if(glfwGetKey(window,GLFW_KEY_S) == GLFW_PRESS)
    {
        _position -= 0.1f*_zAxis;
    }
    if(glfwGetKey(window,GLFW_KEY_W) == GLFW_PRESS)
    {
        _position += 0.1f*_zAxis;
    }

    if(glfwGetKey(window, GLFW_KEY_UP))
    {
        _pitch += 0.01f;
    }
    if(glfwGetKey(window, GLFW_KEY_DOWN))
    {
        _pitch -= 0.01f;
    }
    if(glfwGetKey(window, GLFW_KEY_RIGHT))
    {
        _yaw -= 0.01f;
    }
    if(glfwGetKey(window, GLFW_KEY_LEFT))
    {
        _yaw += 0.01f;
    }

    Vector3float front(cos(_yaw)*cos(_pitch),
                       sin(_pitch),
                       sin(_yaw)*cos(_pitch));

    _zAxis = front;
    _xAxis = Math::normalize(Vector3float(_zAxis.z,0,-_zAxis.x));
    _yAxis = Math::cross(_zAxis, _xAxis);
}