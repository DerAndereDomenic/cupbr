#ifndef CUPBR_RENDERER_PBRENDERER_CUH
#define CUPBR_RENDERER_PBRENDERER_CUH

#include <memory>
#include <DataStructure/Image.cuh>
#include <Scene/Scene.cuh>
#include <DataStructure/Camera.cuh>

namespace cupbr
{
    /**
    *   @brief The different rendering methods
    */
    enum class RenderingMethod
    {
        RAYTRACER,
        WHITTED,
        PATHTRACER,
        METROPOLIS,
        GRADIENTDOMAIN,
        VOLUME
    };

    /**
    *   @brief A class to model the physically based renderer
    */
    class PBRenderer
    {
        public:
        /**
        *   @brief Create the renderer
        *   @param[in] method The rendering method (default = RAYTRACER)
        */
        PBRenderer(const RenderingMethod& method = RenderingMethod::RAYTRACER);

        /**
        *   @brief Destructor
        */
        ~PBRenderer();

        /**
        *   @brief Set the image output size
        *   @param[in] width The output width
        *   @param[in] height The output height
        */
        void setOutputSize(const uint32_t& width, const uint32_t& height);

        /**
        *   @brief Register the scene to render
        *   @param[in] scene The scene to render
        */
        void registerScene(Scene* scene);

        /**
        *   @brief Set the rendering method
        *   @param[in] method The new method
        */
        void setMethod(const RenderingMethod& method);

        /**
        *   @brief Render the scene
        *   @param[in] camera The camera
        */
        void render(Camera* camera);

        /**
        *   @brief The output generated by render()
        *   @return The HDR image produced by the renderer
        */
        Image<Vector3float>* getOutputImage();

        /**
        *   @brief The estimate of x gradients
        *   @return The x gradient image
        */
        Image<Vector3float>* getGradientX();

        /**
        *   @brief The estimate of y gradients
        *   @return The y gradient image
        */
        Image<Vector3float>* getGradientY();

        /**
        *   @brief Get the currently selected rendering method
        *   @return The rendering method
        */
        RenderingMethod getMethod();

        /**
        *   @brief Reset the current render
        */
        void reset();

        private:
        class Impl;
        std::unique_ptr<Impl> impl;     /**< The implementation pointer */
    };
} //namespace cupbr

#endif