#include <iostream>

#include <GL/GLRenderer.h>
#include <string>

#include <Core/Memory.h>

#include <imgui.h>

namespace cupbr
{
    class GLRenderer::Impl
    {
        public:
        Impl(const uint32_t& width, const uint32_t& height);

        ~Impl();

        /**
        *   @brief Create the screen texture
        *   @param[in] width The framebuffer width
        *   @param[in] height The framebuffer height
        */
        void createGLTexture(const uint32_t& width, const uint32_t& height);

        uint32_t _screen_texture;               /**< The screen quad texture */

        cudaGraphicsResource* _cuda_resource;   /**< CUDA resource */
        cudaArray* _texture_ptr;                /**< Texture pointer */

        float _viewport_width = 0;
        float _viewport_height = 0;

        float _viewport_x = 0;
        float _viewport_y = 0;
    };

    GLRenderer::Impl::Impl(const uint32_t& width, const uint32_t& height)
    {
        createGLTexture(width, height);
    }

    GLRenderer::Impl::~Impl()
    {
        glDeleteTextures(1, &_screen_texture);
    }

    GLRenderer::GLRenderer(const uint32_t& width, const uint32_t& height)
    {
        impl = std::make_unique<Impl>(width, height);
    }

    GLRenderer::~GLRenderer() = default;

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

    bool
    GLRenderer::displayImage(const RenderBuffer& img)
    {
        //Deprecated
        //cudaSafeCall(cudaMemcpyToArray(impl->_texture_ptr, 0, 0, img.data(), 4 * img.size(), cudaMemcpyDeviceToDevice));
        cudaSafeCall(cudaMemcpy2DToArray(impl->_texture_ptr, 0, 0, img.data(), img.width() * sizeof(Vector4uint8_t), img.width() * sizeof(Vector4uint8_t), img.height(), cudaMemcpyDeviceToDevice));

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::Begin("Viewport");
        ImVec2 viewport_panel_size = ImGui::GetContentRegionAvail();
        ImVec2 viewport_position = ImGui::GetWindowPos();

        impl->_viewport_x = viewport_position.x;
        impl->_viewport_y = viewport_position.y;

        bool resized = false;
        if(viewport_panel_size.x != impl->_viewport_width || 
           viewport_panel_size.y != impl->_viewport_height)
        {
            impl->_viewport_width = viewport_panel_size.x;
            impl->_viewport_height = viewport_panel_size.y;
            resized = true;
        }

        ImGui::Image((void*)(impl->_screen_texture), ImVec2(img.width(), img.height()), ImVec2(0, 1), ImVec2(1, 0));

        ImGui::End();
        ImGui::PopStyleVar();

        //glDrawArrays(GL_TRIANGLES, 0, 6);
        return resized;
    }

    void 
    GLRenderer::clear() const
    {
        glClear(GL_COLOR_BUFFER_BIT);
    }

    void 
    GLRenderer::setViewport(uint32_t x, uint32_t y, uint32_t width, uint32_t height) const
    {
        glViewport(x, y, width, height);
    }

    void 
    GLRenderer::onResize(const uint32_t& width, const uint32_t& height)
    {
        cudaSafeCall(cudaGraphicsUnmapResources(1, &(impl->_cuda_resource), 0));
        cudaSafeCall(cudaGraphicsUnregisterResource(impl->_cuda_resource));

        glDeleteTextures(1, &(impl->_screen_texture));

        impl->createGLTexture(width, height);
    }

    Vector2float 
    GLRenderer::getViewportSize() const
    {
        return Vector2float(impl->_viewport_width, impl->_viewport_height);
    }

    Vector2float 
    GLRenderer::getViewportPosition() const
    {
        return Vector2float(impl->_viewport_x, impl->_viewport_y);
    }

} //namespace cupbr