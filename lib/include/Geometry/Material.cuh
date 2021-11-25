#ifndef __CUPBR_GEOMETRY_MATERIAL_CUH
#define __CUPBR_GEOMETRY_MATERIAL_CUH

#include <Core/CUDA.cuh>
#include <Math/Vector.h>

namespace cupbr
{

    /**
    *   @brief The types of materials supported
    */
    enum class MaterialType
    {
        LAMBERT,
        PHONG,
        MIRROR,
        GLASS,
        GGX
    };

    /**
    *   @brief A class to model different materials
    */
    class Material
    {
        public:
        /**
        *   @brief Default constructor
        */
        Material() = default;

        Vector3float albedo_e = Vector3float(0);                    /**< The emissive albedo */
        Vector3float albedo_d = Vector3float(1);                    /**< The diffuse albedo */
        Vector3float albedo_s = Vector3float(0);                    /**< The specular albedo */
        float shininess = 128.0f * 0.4f;                            /**< The object shininess */
        float roughness = 1.0f;                                     /**< The material roughness for ggx */
        float eta = 1.5f;                                           /**< Index of refraction */

        MaterialType type = MaterialType::LAMBERT;                  /**< The material type */

        /**
        *   @brief Compute the brdf of the material
        *   @param[in] position The position
        *   @param[in] inc_dir The incoming direction
        *   @param[in] out_dir The outgoing directio
        *   @return The brdf
        *   @pre dot(inc_dir,out_dir) >= 0
        */
        __host__ __device__
        Vector3float brdf(const Vector3float& position, const Vector3float& inc_dir, const Vector3float& out_dir, const Vector3float& normal);

        /**
        *   @brief Importance sample the material
        *   @param[in] seed The seed for the rng
        *   @param[in] inc_dir The incoming direction
        *   @param[in] normal The surface normal
        *   @return A 4D Vector. The first three components mark the new direction and the w component the sampling probability
        *   @note For glass the w component is 0 for reflection and 1 for refraction
        */
        __host__ __device__
        Vector4float sampleDirection(uint32_t& seed, const Vector3float& inc_dir, const Vector3float& normal);

        /**
        *   \brief Henyey greenstein phase function
        *   \param[in] g The scattering parameter
        *   \param[in] cos_theta The angle between incoming and outgoing direction
        *   \return The scattering probability
        */
        __host__ __device__
        static float henyeyGreensteinPhaseFunction(const float& g, const float& cos_theta);

        /**
        *   \brief Sample the henyey greenstein phase function  
        *   \param[in] g The scattering parameter
        *   \param[in] forward The forward scattering direction 
        *   \param[in] seed The seed
        *   \return A 4D vector with the direction and pdf as 4th component
        */
        __host__ __device__
        static Vector4float sampleHenyeyGreensteinPhaseFunction(const float& g, const Vector3float& forward, uint32_t& seed);

        private:
        /**
        *   @brief Compute the lambert brdf
        *   @param[in] position The position
        *   @param[in] inc_dir The incoming direction
        *   @param[in] out_dir The outgoing directio
        *   @return The brdf
        *   @pre dot(inc_dir,out_dir) >= 0
        */
        __host__ __device__
        Vector3float brdf_lambert();

        /**
        *   @brief Compute the phong brdf
        *   @param[in] position The position
        *   @param[in] inc_dir The incoming direction
        *   @param[in] out_dir The outgoing directio
        *   @return The brdf
        *   @pre dot(inc_dir,out_dir) >= 0
        */
        __host__ __device__
        Vector3float brdf_phong(const Vector3float& position, const Vector3float& inc_dir, const Vector3float& out_dir, const Vector3float& normal);

        /**
        *   @brief Compute the mirror brdf
        *   @param[in] position The position
        *   @param[in] inc_dir The incoming direction
        *   @param[in] out_dir The outgoing directio
        *   @return The brdf
        *   @pre dot(inc_dir,out_dir) >= 0
        */
        __host__ __device__
        Vector3float brdf_mirror(const Vector3float& position, const Vector3float& inc_dir, const Vector3float& out_dir, const Vector3float& normal);

        /**
        *   @brief Compute glass btdf
        *   @param[in] position The position
        *   @param[in] inc_dir The incoming direction
        *   @param[in] out_dir The outgoing directio
        *   @return The btdf
        */
        __host__ __device__
        Vector3float btdf_glass(const Vector3float& position, const Vector3float& inc_dir, const Vector3float& out_dir, const Vector3float& normal);

        /**
         *   @brief Compute GGX brdf
         *   @param[in] position The position
         *   @param[in] inc_dir The incoming direction
         *   @param[in] out_dir The outgoing directio
         *   @return The brdf
         */
        __host__ __device__
        Vector3float brdf_ggx(const Vector3float& position, const Vector3float& inc_dir, const Vector3float& out_dir, const Vector3float& normal);

        /**
        *   @brief Importance sample lambert material
        *   @param[in] seed The seed for the rng
        *   @param[in] normal The surface normal
        *   @return A 4D Vector. The first three components mark the new direction and the w component the sampling probability
        */
        __host__ __device__
        Vector4float sample_lambert(uint32_t& seed, const Vector3float& normal);

        /**
        *   @brief Importance sample mirror material
        *   @param[in] inc_dir The incoming direction
        *   @param[in] normal The surface normal
        *   @return A 4D Vector. The first three components mark the new direction and the w component the sampling probability
        */
        __host__ __device__
        Vector4float sample_mirror(const Vector3float& inc_dir, const Vector3float& normal);

        /**
        *   @brief Importance sample glass material
        *   @param[in] seed The seed for the rng
        *   @param[in] inc_dir The incoming direction
        *   @param[in] normal The surface normal
        *   @return A 4D Vector. The first three components mark the new direction and the w component the sampling probability
        *   @note The w component is 0 for reflection and 1 for refraction
        */
        __host__ __device__
        Vector4float sample_glass(uint32_t& seed, const Vector3float& inc_dir, const Vector3float& normal);

        /**
         *   @brief Importance sample gxx material
         *   @param[in] seed The seed for the rng
         *   @param[in] inc_dir The incoming direction
         *   @param[in] normal The surface normal
         *   @return A 4D Vector. The first three components mark the new direction and the w component the sampling probability
         */
        __host__ __device__
        Vector4float sample_ggx(uint32_t& seed, const Vector3float& inc_dir, const Vector3float& normal);
    };

} //namespace cupbr

#include "../../src/Geometry/MaterialDetail.cuh"

#endif