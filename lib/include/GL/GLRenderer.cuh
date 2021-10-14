#ifndef __CUPBR_GL_GLRENDERER_H
#define __CUPBR_GL_GLRENDERER_H

#include <DataStructure/RenderBuffer.cuh>

#include <memory>
#include <glad/glad.h>

#include <cuda_gl_interop.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace cupbr
{
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
        void renderTexture(const RenderBuffer& img);

        private:
        class Impl;
        std::unique_ptr<Impl> impl;     /**< Implementation pointer */
    };

} //namespace cupbr

#endif