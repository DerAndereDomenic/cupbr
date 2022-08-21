#include <DataStructure/Camera.h>
#include <Math/Functions.h>
#include <GLFW/glfw3.h>

namespace cupbr
{
    Camera::Camera(const float& aspect_ratio)
    {
        onResize(aspect_ratio);
    }

    void
    Camera::processInput(Window* window, const float& delta_time)
    {
        _moved = false;
        float speed = 4.0f;
        //Keyboard
        if (glfwGetKey(static_cast<GLFWwindow*>(window->getInternalWindow()), Key::LeftShift) == GLFW_PRESS)
        {
            speed *= 4;
        }
        if (glfwGetKey(static_cast<GLFWwindow*>(window->getInternalWindow()), Key::LeftControl) == GLFW_PRESS)
        {
            speed /= 4;
        }
        if (glfwGetKey(static_cast<GLFWwindow*>(window->getInternalWindow()), Key::D) == GLFW_PRESS)
        {
            _position += speed * _xAxis * delta_time;
            _moved = true;
        }
        if (glfwGetKey(static_cast<GLFWwindow*>(window->getInternalWindow()), Key::A) == GLFW_PRESS)
        {
            _position -= speed * _xAxis * delta_time;
            _moved = true;
        }
        if (glfwGetKey(static_cast<GLFWwindow*>(window->getInternalWindow()), Key::S) == GLFW_PRESS)
        {
            _position -= speed * _zAxis * delta_time;
            _moved = true;
        }
        if (glfwGetKey(static_cast<GLFWwindow*>(window->getInternalWindow()), Key::W) == GLFW_PRESS)
        {
            _position += speed * _zAxis * delta_time;
            _moved = true;
        }


        //Mouse
        Vector2float mouse_pos = window->getMousePosition();

        if (_firstMouse)
        {
            _lastX = static_cast<float>(mouse_pos.x);
            _lastY = static_cast<float>(mouse_pos.y);
            _firstMouse = false;
        }

        float sensitivity = 0.002f;
        float xoffset = _lastX - static_cast<float>(mouse_pos.x);
        float yoffset = _lastY - static_cast<float>(mouse_pos.y);

        if (!(Math::safeFloatEqual(xoffset, 0.0f) && Math::safeFloatEqual(yoffset, 0.0f)))
        {
            _moved = true;
        }

        _lastX = static_cast<float>(mouse_pos.x);
        _lastY = static_cast<float>(mouse_pos.y);

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

        lookAt(_position + front);
    }

    void
    Camera::stop(Window* window)
    {
        Vector2float mouse_pos = window->getMousePosition();
        _lastX = mouse_pos.x;
        _lastY = mouse_pos.y;
        _moved = false;
    }

    void 
    Camera::lookAt(const Vector3float& target)
    {
        Vector3float direction = Math::normalize(target - _position);
        _zAxis = direction;
        _xAxis = _aspect_ratio * Math::normalize(Vector3float(_zAxis.z, 0, -_zAxis.x));
        _yAxis = Math::cross(_zAxis, _xAxis) / _aspect_ratio;
    }

    void
    Camera::onResize(const float& aspect_ratio)
    {
        _aspect_ratio = aspect_ratio;

        _xAxis = _aspect_ratio * Math::normalize(_xAxis);
    }

} //namespace cupbr