#include <iostream>

#include <GL/GLRenderer.cuh>
#include <GL/glew.h>
#include <string>

GLRenderer
GLRenderer::createHostObject()
{
    GLRenderer result;

    std::string vertexCode =
    "#version 330 core\n\
    layout (location=0) in vec2 aPosition;\n\
    \
    void main()\n\
    {\n\
        gl_Position = vec4(aPosition, 0, 1);\n\
    }";

    std::string fragmentCode =
    "#version 330 core\n\
    out vec4 FragColor;\n\
    \
    void main()\n\
    {\n\
        FragColor = vec4(1,1,0,1);\
    }";

    ///////////////////////////////////////////////////////
    ///             Shader Compilation                  ///
    ///////////////////////////////////////////////////////
    uint32_t vertexShader;
    uint32_t fragmentShader;
    int32_t vertexSuccess;
    int32_t fragmentSuccess;

    vertexShader = glCreateShader(GL_VERTEX_SHADER);
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    const char* vertexSource = vertexCode.c_str();
    const char* fragmentSource = fragmentCode.c_str();

    glShaderSource(vertexShader, 1, &vertexSource, NULL);
    glCompileShader(vertexShader);
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &vertexSuccess);

    glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
    glCompileShader(fragmentShader);
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &fragmentSuccess);

    if(!vertexSuccess)
    {
        int32_t length;
        glGetShaderiv(vertexShader, GL_INFO_LOG_LENGTH, &length);
        char* infoLog = (char*)malloc(sizeof(char) * length);
        glGetShaderInfoLog(vertexShader, length, &length, infoLog);
        std::cout << infoLog << std::endl;
        free(infoLog);
    }

    if(!fragmentSuccess)
    {
        int32_t length;
        glGetShaderiv(fragmentShader, GL_INFO_LOG_LENGTH, &length);
        char* infoLog = (char*)malloc(sizeof(char) * length);
        glGetShaderInfoLog(fragmentShader, length, &length, infoLog);
        std::cout << infoLog << std::endl;
        free(infoLog);
    }

    return result;
}

void
GLRenderer::destroyHostObject(GLRenderer& object)
{

}

void
GLRenderer::renderTexture(const RenderBuffer& img)
{
    
}