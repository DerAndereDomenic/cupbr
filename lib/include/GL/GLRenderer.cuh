#ifndef __CUPBR_GL_GLRENDERER_H
#define __CUPBR_GL_GLRENDERER_H

#include <DataStructure/RenderBuffer.cuh>
#include <GL/glew.h>

#include <cuda_gl_interop.h>

class GLRenderer
{
    public:
        GLRenderer() = default;

        static GLRenderer
        createHostObject(const uint32_t& width, const uint32_t& height);

        static void
        destroyHostObject(GLRenderer& object);

        void
        renderTexture(const RenderBuffer& img);
    private:
        void createShader();

        void createQuadVBO();

        void createGLTexture(const uint32_t& width, const uint32_t& height);

        uint32_t _vbo;
        uint32_t _shader;
        uint32_t _screen_texture;

        cudaGraphicsResource* _cuda_resource;
        cudaArray* _texture_ptr;
};

#endif