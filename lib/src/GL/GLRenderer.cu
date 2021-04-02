#include <iostream>

#include <GL/GLRenderer.cuh>
#include <GL/glew.h>
#include <string>

GLRenderer
GLRenderer::createHostObject(const uint32_t& width, const uint32_t& height)
{
    GLRenderer result;

    std::string vertexCode =
    "#version 330 core\n\
    layout (location=0) in vec2 aPosition;\n\
    layout (location=1) in vec2 aTexture;\n\
    \
    out vec2 frag_tex;\n\
    \
    void main()\n\
    {\n\
        gl_Position = vec4(aPosition, 0, 1);\n\
        frag_tex = aTexture;\n\
    }";

    std::string fragmentCode =
    "#version 330 core\n\
    out vec4 FragColor;\n\
    \
    in vec2 frag_tex;\n\
    \
    void main()\n\
    {\n\
        FragColor = vec4(frag_tex,0,1);\
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
    float vertices[24] = 
    {
        -1,-1, 0, 0,
        1,-1, 1, 0,
        1,1, 1, 1,

        -1,-1, 0, 0,
        1,1, 1, 1,
        -1,1, 0, 1
    };

    glGenBuffers(1, &result._vbo);
    glBindBuffer(GL_ARRAY_BUFFER, result._vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)(sizeof(float)*2));
    glEnableVertexAttribArray(1);

    ///////////////////////////////////////////////////////
    ///             Screen Texture                      ///
    ///////////////////////////////////////////////////////
    glGenTextures(1, &result._screen_texture);
    glBindTexture(GL_TEXTURE_2D, result._screen_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenerateMipmap(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    return result;
}

void
GLRenderer::destroyHostObject(GLRenderer& object)
{
    glDeleteProgram(object._shader);
    glDeleteVertexArrays(1, &object._vbo);
    glDeleteTextures(1, &object._screen_texture);
}

void
GLRenderer::renderTexture(const RenderBuffer& img)
{
    glDrawArrays(GL_TRIANGLES, 0, 6);
}