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

    if(glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
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

    if(glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
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

    if(glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
    {
        float alpha = -0.01f;
        float c = cosf(alpha);
        float s = sinf(alpha);

        float y = c*_zAxis.y - s*_zAxis.z;
        float x = _zAxis.x;
        float z = s*_zAxis.y + c*_zAxis.z;

        _zAxis = Vector3float(x,y,z);

        y = c*_yAxis.y - s*_yAxis.z;
        x = _yAxis.x;
        z = s*_yAxis.y + c*_yAxis.z;

        _yAxis = Vector3float(x,y,z);
    }

    if(glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
    {
        float alpha = 0.01f;
        float c = cosf(alpha);
        float s = sinf(alpha);

        float y = c*_zAxis.y - s*_zAxis.z;
        float x = _zAxis.x;
        float z = s*_zAxis.y + c*_zAxis.z;

        _zAxis = Vector3float(x,y,z);

        y = c*_yAxis.y - s*_yAxis.z;
        x = _yAxis.x;
        z = s*_yAxis.y + c*_yAxis.z;

        _yAxis = Vector3float(x,y,z);
    }
}