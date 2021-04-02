#ifndef __CUPBR_GL_GLRENDERER_H
#define __CUPBR_GL_GLRENDERER_H

#include <DataStructure/RenderBuffer.cuh>

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
};

#endif