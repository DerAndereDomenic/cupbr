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

    ///////////////////////////////////////////////////////
    ///             Shader Linking                      ///
    ///////////////////////////////////////////////////////
    int32_t success;

    result._shader = glCreateProgram();
    glAttachShader(result._shader, vertexShader);
    glAttachShader(result._shader, fragmentShader);
    glLinkProgram(result._shader);

    glGetProgramiv(result._shader, GL_LINK_STATUS, &success);

    if(!success)
    {
        int32_t length;
		glGetProgramiv(result._shader, GL_INFO_LOG_LENGTH, &length);
		char* infoLog = (char*)malloc(sizeof(char) * length);
		glGetProgramInfoLog(result._shader, length, &length, infoLog);
		std::cout << infoLog << std::endl;
		free(infoLog);

    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    //Bind
    glUseProgram(result._shader);

    ///////////////////////////////////////////////////////
    ///             Vertex Buffer                       ///
    ///////////////////////////////////////////////////////
    float vertices[6] = 
    {
        -0.5f,-0.5f,
        0.5f,-0.5f,
        0.0f,0.5f
    };

    uint32_t vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    return result;
}

void
GLRenderer::destroyHostObject(GLRenderer& object)
{
    glDeleteProgram(object._shader);
}

void
GLRenderer::renderTexture(const RenderBuffer& img)
{
    
}