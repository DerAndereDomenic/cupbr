#include <DataStructure/Camera.cuh>
#include <Math/Functions.cuh>

void
Camera::processInput(GLFWwindow* window)
{
    _moved = false;
    //Keyboard
    if(glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    {
        _position += 0.1f*_xAxis;
        _moved = true;
    }
    if(glfwGetKey(window,GLFW_KEY_A) == GLFW_PRESS)
    {
        _position -= 0.1f*_xAxis;
        _moved = true;
    }
    if(glfwGetKey(window,GLFW_KEY_S) == GLFW_PRESS)
    {
        _position -= 0.1f*_zAxis;
        _moved = true;
    }
    if(glfwGetKey(window,GLFW_KEY_W) == GLFW_PRESS)
    {
        _position += 0.1f*_zAxis;
        _moved = true;
    }


    //Mouse
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);

    if(_firstMouse)
    {
        _lastX = xpos;
        _lastY = ypos;
        _firstMouse = false;
    }

    float sensitivity = 0.002f;
    float xoffset = _lastX - xpos;
    float yoffset = _lastY - ypos;

    if(!(Math::safeFloatEqual(xoffset,0.0f) && Math::safeFloatEqual(yoffset,0.0f)))
    {
        _moved = true;
    }

    _lastX = xpos;
    _lastY = ypos;

    xoffset *= sensitivity;
    yoffset *= sensitivity;

    _yaw += xoffset;
    _pitch += yoffset;

    if(_pitch > 3.14159f/2.0f)
        _pitch = 3.14159f/2.0f;
    if(_pitch < -3.14159f/2.0f)
        _pitch = -3.14159f/2.0f;

    Vector3float front(cos(_yaw)*cos(_pitch),
                       sin(_pitch),
                       sin(_yaw)*cos(_pitch));

    _zAxis = front;
    _xAxis = Math::normalize(Vector3float(_zAxis.z,0,-_zAxis.x));
    _yAxis = Math::cross(_zAxis, _xAxis);
}