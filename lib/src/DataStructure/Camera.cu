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

    if(glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_1) == GLFW_PRESS)
    {
        float alpha = 0.01f;
        float c = cosf(alpha);
        float s = sinf(alpha);

        float x = c*_zAxis.x + s*_zAxis.z;
        float y = _zAxis.y;
        float z = -s*_zAxis.x + c*_zAxis.z;

        _zAxis = Vector3float(x,y,z);

        x = c*_xAxis.x + s*_xAxis.z;
        y = _xAxis.y;
        z = -s*_xAxis.x + c*_xAxis.z;

        _xAxis = Vector3float(x,y,z);
    }

    if(glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_2) == GLFW_PRESS)
    {
        float alpha = -0.01f;
        float c = cosf(alpha);
        float s = sinf(alpha);

        float x = c*_zAxis.x + s*_zAxis.z;
        float y = _zAxis.y;
        float z = -s*_zAxis.x + c*_zAxis.z;

        _zAxis = Vector3float(x,y,z);

        x = c*_xAxis.x + s*_xAxis.z;
        y = _xAxis.y;
        z = -s*_xAxis.x + c*_xAxis.z;

        _xAxis = Vector3float(x,y,z);
    }
}