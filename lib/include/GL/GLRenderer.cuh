#ifndef __CUPBR_GL_GLRENDERER_H
#define __CUPBR_GL_GLRENDERER_H

#include <DataStructure/RenderBuffer.cuh>

class GLRenderer
{
    public:
        GLRenderer() = default;

        static GLRenderer
        createHostObject();

        static void
        destroyHostObject(GLRenderer& object);

        void
        renderTexture(const RenderBuffer& img);
    private:
        uint32_t _shader;
        uint32_t _screen_texture;
};

#endif