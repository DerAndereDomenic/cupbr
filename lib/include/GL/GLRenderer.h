#ifndef __CUPBR_GL_GLRENDERER_H
#define __CUPBR_GL_GLRENDERER_H

#include <DataStructure/RenderBuffer.h>

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
        *   @return True if the viewport was resized
        */
        bool displayImage(const RenderBuffer& img);

        /**
        *   @brief Clear the window
        */
        void clear() const;

        /**
        *   @brief Resize the viewport
        *   @param[in] x The x position
        *   @param[in] y The y position
        *   @param[in] width The width
        *   @param[in] height The height
        */
        void setViewport(uint32_t x, uint32_t y, uint32_t width, uint32_t height) const;

        /**
        *   @brief This gets called if the window gets resized
        *   @param[in] width The new width
        *   @param[in] height The new height
        */
        void onResize(const uint32_t& width, const uint32_t& height);

        /**
        *   @brief Get the size of the viewport
        *   @return Vector containing (width, height)
        */
        Vector2float getViewportSize() const;

        private:
        class Impl;
        std::unique_ptr<Impl> impl;     /**< Implementation pointer */
    };

} //namespace cupbr

#endif