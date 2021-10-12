#ifndef __CUPBR_GL_WINDOW_H
#define __CUPBR_GL_WINDOW_H

#include <cstdint>

struct GLFWwindow;

namespace cupbr
{
    /**
    *   @brief A class to model a window 
    */
    class Window
    {
        public:

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
            *   @brief Get the internal window handle
            *   @return The window handle 
            */
            inline void* getInternalWindow() {return _internal_window;}

            /**
            *   @brief Get the window width
            *   @return The width 
            */
            inline uint32_t width() const {return _width;}

            /**
            *   @brief Get the window height
            *   @return The height 
            */
            inline uint32_t height() const {return _height;}

        private:
            GLFWwindow* _internal_window;       /**< The internal window */
            uint32_t _width;                    /**< The window width */
            uint32_t _height;                   /**< The window height */
    };

} //namespace cupbr

#endif
