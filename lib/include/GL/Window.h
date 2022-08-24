#ifndef __CUPBR_GL_WINDOW_H
#define __CUPBR_GL_WINDOW_H

#include <cstdint>
#include <functional>
#include <Core/Event.h>
#include <Math/Vector.h>
#include <Core/KeyCodes.h>
#include <GL/GLRenderer.h>

struct GLFWwindow;

namespace cupbr
{
    /**
    *   @brief A class to model a window
    */
    class Window
    {
        public:
        using EventCallbackFn = std::function<void(Event&)>;
        /**
        *   @brief Default constructor
        */
        Window() = default;

        /**
        *   @brief Constructor
        *   @param[in] title The title
        *   @param[in] width The width
        *   @param[in] height The height
        */
        Window(const char* title, const uint32_t& width, const uint32_t& height);

        /**
        *   @brief Destructor
        */
        ~Window();

        /**
        *   @brief Begin a new imgui frame
        */
        void imguiBegin();

        /**
        *   @brief End the imgui frame
        */
        void imguiEnd();

        /**
        *   @brief Polls events and swaps gl buffers
        */
        void spinOnce();

        /**
        *   @brief Get the position of the mouse
        *   @return The mouse position
        */
        Vector2float getMousePosition() const;

        /**
        *   @brief Change between view and edit mode
        *   @param[in] edit_mode If edit mode is enabled or not
        */
        void setEditMode(const bool edit_mode);

        /**
        *   @brief Check if the window should be closed by some external event
        *   @return True if the window should be closed
        */
        bool shouldClose() const;

        /**
        *   @brief Close the window
        */
        void close();

        /**
        *   @brief Gets called if the window gets resized
        *   @param[in] width The new width
        *   @param[in] height The new height
        */
        void onResize(const float& width, const float& height);

        /**
        *   @brief Display an image
        *   @param[in] img The image to display
        */
        void displayImage(const RenderBuffer& img);

        /**
        * @brief Get the Window Position
        * @return Vector (x,y)
        */
        Vector2float getWindowPosition() const;

        /**
        *   @brief Set the viewport
        *   @param[in] x The x position
        *   @param[in] y The y position
        *   @param[in] width The width
        *   @param[in] height The height
        */
        inline void setViewport(const uint32_t& x, const uint32_t& y, const uint32_t& width, const uint32_t& height) { _renderer->setViewport(x, y, width, height); }

        /**
        *   @brief Clear the viewport
        */
        inline void clear() const { _renderer->clear(); }

        /**
        *   @brief Get the size of the current view port
        *   @return Vector of (width, height)
        */
        inline Vector2float getViewportSize() const { return _renderer->getViewportSize(); }

        /**
        * @brief Get the positition of the current view port
        * @return Vector of (x,y)
        */
        inline Vector2float getViewportPosition() const { return _renderer->getViewportPosition(); }

        /**
        *   @brief Set the event callback
        *   @param[in] callback The callback function called on events
        */
        inline void setEventCallback(const EventCallbackFn& callback) { _event_callback = callback; }

        /**
        *   @brief Get the internal window handle
        *   @return The window handle
        */
        inline void* getInternalWindow() { return _internal_window; }

        /**
        *   @brief Get the window width
        *   @return The width
        */
        inline uint32_t width() const { return _width; }

        /**
        *   @brief Get the window height
        *   @return The height
        */
        inline uint32_t height() const { return _height; }

        /**
        *   @brief Get the delta time
        *   @return delta_time in s
        */
        inline float delta_time() const { return _delta_time; }

        private:
        void setDarkThemeColors();
        GLFWwindow* _internal_window;           /**< The internal window */
        uint32_t _width;                        /**< The window width */
        uint32_t _height;                       /**< The window height */
        EventCallbackFn _event_callback;        /**< The event callback */
        float _delta_time = 0;                  /**< The delta time */
        float _last_time = 0;                   /**< The last time events were polled */
        std::unique_ptr<GLRenderer> _renderer;  /**< The internal GL renderer class */
    };

} //namespace cupbr

#endif
