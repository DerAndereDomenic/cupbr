#ifndef __CUPBR_GL_GLRENDERER_H
#define __CUPBR_GL_GLRENDERER_H

#include <DataStructure/RenderBuffer.cuh>
#include <GL/glew.h>

#include <memory>

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
        */
        GLRenderer(const uint32_t& width, const uint32_t& height);

        /**
        *   @brief Destroy the rendering context
        */
        ~GLRenderer();

        /**
        *   @brief Render a device image on a quad
        *   @param[in] img The device image 
        */
        void
        renderTexture(const RenderBuffer& img);
    private:
        class Impl;
        std::unique_ptr<Impl> impl;     /**< Implementation pointer */
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