#include <DataStructure/Camera.h>
#include <Math/Functions.h>

namespace cupbr
{
    Camera::Camera(const uint32_t& width, const uint32_t& height)
    {
        _aspect_ratio = static_cast<float>(width) / static_cast<float>(height);

        _xAxis = Vector3float(_aspect_ratio, 0, 0);
    }

    void
    Camera::processInput(GLFWwindow* window, const float& delta_time)
    {
        _moved = false;
        //Keyboard
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        {
            _position += 4.0f * _xAxis * delta_time;
            _moved = true;
        }
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        {
            _position -= 4.0f * _xAxis * delta_time;
            _moved = true;
        }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        {
            _position -= 4.0f * _zAxis * delta_time;
            _moved = true;
        }
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        {
            _position += 4.0f * _zAxis * delta_time;
            _moved = true;
        }


        //Mouse
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);

        if (_firstMouse)
        {
            _lastX = static_cast<float>(xpos);
            _lastY = static_cast<float>(ypos);
            _firstMouse = false;
        }

        float sensitivity = 0.002f;
        float xoffset = _lastX - static_cast<float>(xpos);
        float yoffset = _lastY - static_cast<float>(ypos);

        if (!(Math::safeFloatEqual(xoffset, 0.0f) && Math::safeFloatEqual(yoffset, 0.0f)))
        {
            _moved = true;
        }

        _lastX = static_cast<float>(xpos);
        _lastY = static_cast<float>(ypos);

        xoffset *= sensitivity;
        yoffset *= sensitivity;

        _yaw += xoffset;
        _pitch += yoffset;

        if (_pitch > 3.14159f / 2.0f)
            _pitch = 3.14159f / 2.0f;
        if (_pitch < -3.14159f / 2.0f)
            _pitch = -3.14159f / 2.0f;

        Vector3float front(cos(_yaw) * cos(_pitch),
                           sin(_pitch),
                           sin(_yaw) * cos(_pitch));

        _zAxis = front;
        _xAxis = _aspect_ratio * Math::normalize(Vector3float(_zAxis.z, 0, -_zAxis.x));
        _yAxis = Math::cross(_zAxis, _xAxis) / _aspect_ratio;
    }
} //namespace cupbr