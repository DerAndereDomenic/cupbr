#ifndef __CUPBR_INTERACTION_INTERACTOR_CUH
#define __CUPBR_INTERACTION_INTERACTOR_CUH

#include <GL/Window.h>

#include <memory>

#include <DataStructure/Camera.cuh>
#include <Scene/Scene.cuh>
#include <Renderer/PBRenderer.cuh>
#include <Renderer/ToneMapper.cuh>

namespace cupbr
{
    /**
    *   @brief A class to handle user interactions
    */
    class Interactor
    {
    public:
        /**
        *   @brief Cosntructor
        *   @param[in] method The rendering method
        */
        Interactor(const RenderingMethod& method);

        /**
        *   @brief Default destructor
        */
        ~Interactor();

        /**
        *   @brief Add the window for which the input should be handled
        *   @param[in] window The window
        */
        void
            registerWindow(const Window& window);

        /**
        *   @brief Add the scene we want to interact with
        *   @param[in] scene The scene
        */
        void
            registerScene(Scene* scene);

        /**
        *   @brief Add the camera
        *   @param[in] camera The camera
        */
        void
            registerCamera(const Camera& camera);

        /**
        *   @brief This handles the user interaction. It should be called every frame
        */
        void
            handleInteraction();

        /**
        *   @brief If the interaction updated the scene
        *   @return True if a scene element changed
        */
        bool updated();

        /**
        *   @brief Get the tone mapping type
        *   @return The tone mapping type
        */
        ToneMappingType
            getToneMapping();

        /**
        *   @brief Get the rendering method
        *   @return The rendering method
        */
        RenderingMethod
            getRenderingMethod();

        /**
        *   @brief Get the exposure level set by the user
        *   @return The exposure time
        */
        float
            getExposure();

        /**
        *   @brief If post processing should be used
        *   @return True if bloom should be activated
        */
        bool
            usePostProcessing();

        /**
        *   @brief Get the quadratic thresholding curve for bloom
        *   @return The vector containing the curve (threshold, knee - threshold, 2*knee, 0.25/knee)
        */
        Vector4float
            getThreshold();
    private:
        class Impl;
        std::unique_ptr<Impl> impl;     /**< Implementation pointer */
    };

} //namespace cupbr

#endif