#ifndef __CUPBR_INTERACTION_INTERACTOR_H
#define __CUPBR_INTERACTION_INTERACTOR_H

#include <GL/Window.h>

#include <memory>

#include <Core/Event.h>
#include <DataStructure/Camera.h>
#include <Scene/Scene.h>
#include <Renderer/PBRenderer.h>
#include <Renderer/ToneMapper.h>
#include <PostProcessing/PostProcessor.h>

namespace cupbr
{
    /**
    *   @brief A class to handle user interactions
    */
    class Interactor
    {
        public:
        /**
        *   @brief Constructor
        */
        Interactor();

        /**
        *   @brief Default destructor
        */
        ~Interactor();

        /**
        *   @brief Add the window for which the input should be handled
        *   @param[in] window The window
        *   @param[in] menu_width The width of the menu
        */
        void registerWindow(Window* window, const int32_t& menu_width);

        /**
        *   @brief Add the scene we want to interact with
        *   @param[in] scene The scene
        */
        void registerScene(Scene* scene);

        /**
        *   @brief Add the camera
        *   @param[in] camera The camera
        */
        void registerCamera(Camera* camera);

        /**
        *   @brief Register a renderer
        *   @param[in] renderer The renderer
        */
        void registerRenderer(PBRenderer* renderer);

        /**
        *   @brief Register a tone mapper
        *   @param[in] mapper The tone mapper
        */
        void registerToneMapper(ToneMapper* mapper);

        /**
        *   @brief Register post processor
        *   @param[in] post_processor The post processor
        */
        void registerPostProcessor(PostProcessor* post_processor);

        /**
        *   @brief The function called on event
        *   @param[in] event The event to handle
        *   @return True if the event was handled
        */
        bool onEvent(Event& event);

        /**
        *   @brief This handles the user interaction. It should be called every frame
        */
        void handleInteraction();

        /**
        *   @brief If post processing should be used
        *   @return True if post processing is enabled
        */
        bool usePostProcessing();

        /**
        *   @brief Get the quadratic thresholding curve for bloom
        *   @return The vector containing the curve (threshold, knee - threshold, 2*knee, 0.25/knee)
        */
        Vector4float getThreshold();

        /**
        *   @brief If the application should be closed
        *   @return True if escape was pressed
        */
        bool shouldClose();

        /**
        *   @brief Check if a new scene was loaded
        *   @param[out] file_path The path to the new scene
        *   @return True if a new scene was selected
        */
        bool resetScene(std::string& file_path);

        private:
        class Impl;
        std::unique_ptr<Impl> impl;     /**< Implementation pointer */
    };

} //namespace cupbr

#endif