#include <iostream>

#include <GL/GLRenderer.cuh>
#include <string>

#include <Core/Memory.cuh>

class GLRenderer::Impl
{
    public:
        Impl(const uint32_t& width, const uint32_t& height);

        ~Impl();

        /**
        *   @brief Create the screen quad shader 
        */
        void createShader();

        /**
        *   @brief Create the screen quad vbo 
        */
        void createQuadVBO();

        /**
        *   @brief Create the screen texture
        *   @param[in] width The framebuffer width
        *   @param[in] height The framebuffer height 
        */
        void createGLTexture(const uint32_t& width, const uint32_t& height);

        uint32_t _vbo;                          /**< The screen quad vbo */
        uint32_t _shader;                       /**< The screen quad shader */
        uint32_t _screen_texture;               /**< The screen quad texture */

        cudaGraphicsResource* _cuda_resource;   /**< CUDA resource */
        cudaArray* _texture_ptr;                /**< Texture pointer */
};

GLRenderer::Impl::Impl(const uint32_t& width, const uint32_t& height)
{
    createShader();
    createQuadVBO();
    createGLTexture(width, height);
}

GLRenderer::Impl::~Impl()
{
    glDeleteProgram(_shader);
    glDeleteVertexArrays(1, &_vbo);
    glDeleteTextures(1, &_screen_texture);
}

GLRenderer::GLRenderer(const uint32_t& width, const uint32_t& height)
{
    impl = std::make_unique<Impl>(width, height);
}

GLRenderer::~GLRenderer() = default;

void 
GLRenderer::Impl::createShader()
{
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
    uniform sampler2D screen_texture;\n\
    \
    void main()\n\
    {\n\
        FragColor = vec4(texture(screen_texture, frag_tex).rgb ,1);\
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
        char* infoLog = Memory::allocator()->createHostArray<char>(length);
        glGetShaderInfoLog(vertexShader, length, &length, infoLog);
        std::cout << infoLog << std::endl;
        Memory::allocator()->destroyHostArray<char>(infoLog);
    }

    if(!fragmentSuccess)
    {
        int32_t length;
        glGetShaderiv(fragmentShader, GL_INFO_LOG_LENGTH, &length);
        char* infoLog = Memory::allocator()->createHostArray<char>(length);
        glGetShaderInfoLog(fragmentShader, length, &length, infoLog);
        std::cout << infoLog << std::endl;
        Memory::allocator()->destroyHostArray<char>(infoLog);
    }

    ///////////////////////////////////////////////////////
    ///             Shader Linking                      ///
    ///////////////////////////////////////////////////////
    int32_t success;

    _shader = glCreateProgram();
    glAttachShader(_shader, vertexShader);
    glAttachShader(_shader, fragmentShader);
    glLinkProgram(_shader);

    glGetProgramiv(_shader, GL_LINK_STATUS, &success);

    if(!success)
    {
        int32_t length;
		glGetProgramiv(_shader, GL_INFO_LOG_LENGTH, &length);
		char* infoLog = Memory::allocator()->createHostArray<char>(length);
		glGetProgramInfoLog(_shader, length, &length, infoLog);
		std::cout << infoLog << std::endl;
		Memory::allocator()->destroyHostArray<char>(infoLog);
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    //Bind
    glUseProgram(_shader);
}

void 
GLRenderer::Impl::createQuadVBO()
{
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

    glGenBuffers(1, &_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)(sizeof(float)*2));
    glEnableVertexAttribArray(1);
}

void 
GLRenderer::Impl::createGLTexture(const uint32_t& width, const uint32_t& height)
{
    ///////////////////////////////////////////////////////
    ///             Screen Texture                      ///
    ///////////////////////////////////////////////////////
    glGenTextures(1, &_screen_texture);
    glBindTexture(GL_TEXTURE_2D, _screen_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenerateMipmap(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    cudaSafeCall(cudaGraphicsGLRegisterImage(&_cuda_resource, _screen_texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
    cudaSafeCall(cudaGraphicsMapResources(1, &_cuda_resource, 0));
    cudaSafeCall(cudaGraphicsSubResourceGetMappedArray(&_texture_ptr, _cuda_resource, 0, 0));
}

void
GLRenderer::renderTexture(const RenderBuffer& img)
{
    cudaSafeCall(cudaMemcpyToArray(impl->_texture_ptr, 0, 0, img.data(), 4*img.size(), cudaMemcpyDeviceToDevice));

    glDrawArrays(GL_TRIANGLES, 0, 6);
}