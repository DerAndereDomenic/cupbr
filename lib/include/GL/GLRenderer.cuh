#ifndef __CUPBR_GL_GLRENDERER_H
#define __CUPBR_GL_GLRENDERER_H

#include <DataStructure/RenderBuffer.cuh>
#include <GL/glew.h>

#include <cuda_gl_interop.h>

class GLRenderer
{
    public:
        /**
        *   @brief Default constructor 
        */
        GLRenderer() = default;

        /**
        *   @brief Create the rendering context
        *   @param[in] width The width of the framebuffer
        *   @param[in] height The height of the framebuffer
        *   @return The Renderer object 
        */
        static GLRenderer
        createHostObject(const uint32_t& width, const uint32_t& height);

        /**
        *   @brief Destroy the rendering context
        *   @param[in] object The object to be destroyed 
        */
        static void
        destroyHostObject(GLRenderer& object);

        /**
        *   @brief Render a device image on a quad
        *   @param[in] img The device image 
        */
        void
        renderTexture(const RenderBuffer& img);
    private:

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

#endif