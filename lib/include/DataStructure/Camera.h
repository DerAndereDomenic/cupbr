#ifndef __CUPBR_DATASTRUCTURE_CAMERA_H
#define __CUPBR_DATASTRUCTURE_CAMERA_H

#include <Core/CUDA.h>
#include <Math/Vector.h>

#include <GLFW/glfw3.h>

namespace cupbr
{
    /**
    *   @brief A class to model a camera
    */
    class Camera
    {
        public:
        /**
        *   @brief Default constructor
        *   @note This camera expects width == height
        */
        Camera() = default;

        /**
        *   @brief Constructor
        *   @note This camera can be used for different aspect ratios
        */
        Camera(const uint32_t& width, const uint32_t& height);

        /**
        *   @brief Get the world position of the camera
        *   @return The camera position in world space
        */
        __host__ __device__
        Vector3float position() const;

        /**
        *   @brief The x Axis of the camera plane
        *   @return The vector defining the Image plane into x direction in world space
        */
        __host__ __device__
        Vector3float xAxis() const;

        /**
        *   @brief The y Axis of the camera plane
        *   @return The vector defining the Image plane into y direction in world space
        */
        __host__ __device__
        Vector3float yAxis() const;

        /**
        *   @brief The z Axis to the camera plane
        *   @return The vector pointing from the camera position to the image plane in world space
        */
        __host__ __device__
        Vector3float zAxis() const;

        /**
        *   @brief Process keyboard and mouse input
        *   @param[in] window The window
        *   @param[in] delta_time The time between two frames
        */
        void processInput(GLFWwindow* window, const float& delta_time);

        /**
        *   @brief Check if the camera was moved last frame
        *   @return If the camera was moved last frame
        */
        __host__ __device__
        bool moved() const;

        private:
        Vector3float _position = Vector3float(0, 0, 0);     /**< The camera position in world space */
        Vector3float _xAxis = Vector3float(1, 0, 0);        /**< The image plane x Axis in world space */
        Vector3float _yAxis = Vector3float(0, 1, 0);        /**< The image plane y Axis in world space */
        Vector3float _zAxis = Vector3float(0, 0, 1);        /**< The displacement between position and image plane in world space */
        float _aspect_ratio = 1.0f;                         /**< The cameras aspect ratio */

        float _pitch = 0.0f;                                /**< The pitch of the camera */
        float _yaw = 3.14159f / 2.0f;                       /**< The yaw of the camera */

        bool _firstMouse = true;                            /**< Boolean to initialize the first mouse movement */
        float _lastX = 0.0f;                                /**< The last x position of the cursor needed for mouse movement */
        float _lastY = 0.0f;                                /**< The last y position of the cursor needed for mouse movement */

        bool _moved = false;                                /**< If the camera was moved last frame */
    };

} //namespace cupbr

#include "../../src/DataStructure/CameraDetail.h"

#endif