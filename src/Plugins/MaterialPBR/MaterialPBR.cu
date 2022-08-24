#include <CUPBR.h>

namespace cupbr
{

    namespace detail
    {
        CUPBR_HOST_DEVICE
        inline float
        D_GGX(const float& NdotH, const float roughness)
        {
            float a2 = roughness * roughness;
            float d = (NdotH * a2 - NdotH) * NdotH + 1.0f;
            return a2 / (static_cast<float>(M_PI) * d * d);
        }

        CUPBR_HOST_DEVICE
        inline float
        V_SmithJointGGX(const float& NdotL, const float& NdotV, const float& roughness)
        {
            float a2 = roughness * roughness;
            float denomA = NdotV * sqrt(a2 + (1.0f - a2) * NdotL * NdotL);
            float denomB = NdotL * sqrt(a2 + (1.0f - a2) * NdotV * NdotV);

            float G = 2.0f * NdotL * NdotV / (denomA + denomB + EPSILON);

            return G / (4.0f * NdotV * NdotL + EPSILON);
        }

        CUPBR_HOST_DEVICE
        inline Vector3float
        sampleHemisphereCosine( const Vector3float &normal, uint32_t &seed )
        {
            float u   = Math::rnd(seed);
            float r   = sqrtf( u );
            float phi = 2.0f * static_cast<float>(M_PI) * Math::rnd(seed);

            float x = r * cosf(phi);
            float y = r * sinf(phi);
            float z = sqrtf( fmaxf(0.0f, 1.0f - u) );

            return Math::toLocalFrame(normal, Vector3float( x, y, z));
        }

        CUPBR_HOST_DEVICE
        inline Vector3float 
        sampleHemisphereGGX( const Vector3float &inc_dir, const Vector3float &normal, float roughness, uint32_t &seed )
        {
            float u = Math::rnd(seed);
            float v = Math::rnd(seed);

            float cosTheta = sqrtf((1.0f - u) / (1.0f + (roughness * roughness - 1.0f) * u));
            float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
            float phi = 2.0f * static_cast<float>(M_PI) * v;

            float x = sinTheta * cosf(phi);
            float y = sinTheta * sinf(phi);
            float z = cosTheta;

            Vector3float H = Math::toLocalFrame(normal, Vector3float(x, y, z));
            Vector3float L = Math::reflect(inc_dir, H);

            float LdotH = fmaxf(0.0f, Math::dot(inc_dir, H));
            float NdotH = fmaxf(0.0f, Math::dot(normal, H));

            if (Math::dot(normal, L) <= 0.0f)
            {
                return Vector4float(0, 0, 0, 1);
            }

            return L;
        }
    } //namespace detail
    
    class MaterialPBR : public Material
    {
        public:

        MaterialPBR(Properties* properties)
        {
            type = MaterialType::OPAQUE;
            Vector3float base_color = properties->getProperty("base_color", Vector3float(0));
            float metallic = properties->getProperty("metallic", 0.0f);
            float perceptual_roughness = properties->getProperty("perceptual_roughness", 0.0f);

            const float F0 = 0.04f;
            albedo_d = base_color * (1.0f - F0) * (1.0f - metallic);
            albedo_s = (1.0f - metallic) * Vector3float(F0) + (metallic) * base_color;
            roughness = fmaxf(0.001f, perceptual_roughness*perceptual_roughness);
            albedo_e = properties->getProperty("albedo_e", Vector3float(0));
        }

        CUPBR_HOST_DEVICE
        virtual Vector3float
        brdf(const Vector3float& position, const Vector3float& inc_dir, const Vector3float& out_dir, const Vector3float& normal)
        {
            Vector3float diffuse_brdf = albedo_d / static_cast<float>(M_PI);

            Vector3float H = Math::normalize(inc_dir + out_dir);
            float NdotH = fmaxf(0.0f, Math::dot(normal, H));
            float LdotH = fmaxf(0.0f, Math::dot(out_dir, H));
            float NdotV = fmaxf(0.0f, Math::dot(normal, inc_dir));
            float NdotL = fmaxf(0.0f, Math::dot(normal, out_dir));
            float ndf = detail::D_GGX(NdotH, roughness);

            float vis = detail::V_SmithJointGGX(NdotL, NdotV, roughness);
            Vector3float specular_brdf = ndf * vis * Math::fresnel_schlick(albedo_s, LdotH);

            return diffuse_brdf + specular_brdf;
        }

        CUPBR_HOST_DEVICE
        virtual Vector4float
        sampleDirection(uint32_t& seed, const Vector3float& inc_dir, const Vector3float& normal)
        {
            Vector3float direction;
            Vector3float luminance( 0.21263901f, 0.71516868f, 0.07219232f );

            // Sample a diffuse or specular ray?
            float diffuse_weight = Math::dot(luminance, albedo_d);
            float specular_weight = Math::dot(luminance, albedo_s); 
            if (diffuse_weight + specular_weight <= 0)
                return Vector4float(0,0,0,1);

            float diffuse_prob = diffuse_weight / (diffuse_weight + specular_weight);
            if (Math::rnd(seed) < diffuse_prob)
            {
                // Diffuse reflection
                direction = detail::sampleHemisphereCosine( normal, seed );
            }
            else
            {
                // Specular reflection
                direction = detail::sampleHemisphereGGX( inc_dir, normal, roughness, seed );
            }

            // Hint: reject light directions below the horizon
            float NdotL = Math::dot(normal, direction);
            if (NdotL <= 0)
                return Vector4float(0,0,0,1);;

            Vector3float H = Math::normalize(inc_dir + direction);
            float NdotH = fmaxf(0.0f, Math::dot(normal, H));
            float LdotH = fmaxf(0.0f, Math::dot(direction, H));
            float NdotV = fmaxf(0.0f, Math::dot(normal, direction));
            float diffuse_pdf = NdotV / static_cast<float>(M_PI);
            float specular_pdf = detail::D_GGX(NdotH, roughness) * NdotH / fabsf(4.0f * LdotH);;

            // P(ray) = P(ray | diffuse) * P(diffuse) + P(ray | specular) * P(specular)
            float ray_pdf = specular_pdf * (1.0f - diffuse_prob) + diffuse_pdf * diffuse_prob;

            return Vector4float(direction, ray_pdf);
        }

        private:
        Vector3float albedo_d;
        Vector3float albedo_s;
        Vector3float albedo_e;
        float roughness;
    };

    DEFINE_PLUGIN(MaterialPBR, "PBR", "1.0", Material)

}