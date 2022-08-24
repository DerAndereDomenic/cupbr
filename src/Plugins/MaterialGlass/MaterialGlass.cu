#include <CUPBR.h>

namespace cupbr
{
    namespace detail
    {
        CUPBR_HOST_DEVICE
        inline Vector3float
        brdf_mirror(const Vector3float& position, const Vector3float& inc_dir, const Vector3float& out_dir, const Vector3float& normal, const Vector3float& albedo_s)
        {
            Vector3float reflected = Math::reflect(inc_dir, normal);
            return albedo_s * Math::delta(1.0f - Math::dot(out_dir, reflected)) / Math::dot(out_dir, normal);
        }
    } //namespace detail

    class MaterialGlass : public Material
    {
        public:

        MaterialGlass(Properties* properties)
        {
            type = MaterialType::REFRACTIVE;
            albedo_e = properties->getProperty("albedo_e", Vector3float(0));
            albedo_d = properties->getProperty("albedo_d", Vector3float(1));
            albedo_s = properties->getProperty("albedo_s", Vector3float(0));
            eta = properties->getProperty("eta", 1.5f);
        }

        CUPBR_HOST_DEVICE
        virtual Vector3float
        brdf(const Vector3float& position, const Vector3float& inc_dir, const Vector3float& out_dir, const Vector3float& normal)
        {
            if (Math::dot(inc_dir, normal) * Math::dot(out_dir, normal) > 0) //Reflected
            {
                if (Math::dot(inc_dir, normal) > 0)
                {
                    return detail::brdf_mirror(position, inc_dir, out_dir, normal, albedo_s);
                }
                else
                {
                    return detail::brdf_mirror(position, inc_dir, out_dir, -1.0f * normal, albedo_s);
                }

            }
            else
            {
                Vector3float refracted;
                Vector3float n = normal;
                if (Math::dot(inc_dir, normal) > 0)
                {
                    refracted = Math::refract(1.0f / eta, inc_dir, normal);
                }
                else
                {
                    refracted = Math::refract(eta, inc_dir, -1.0f * normal);
                    n = -1.0f * normal;
                }

                return albedo_s * Math::delta(1.0f - Math::dot(refracted, out_dir)) / Math::dot(inc_dir, n);
            }
        }

        CUPBR_HOST_DEVICE
        virtual Vector4float
        sampleDirection(uint32_t& seed, const Vector3float& inc_dir, const Vector3float& n)
        {
            const float NdotV = Math::dot(inc_dir, n);
            bool outside = NdotV > 0.0f;
            float _eta = outside ? 1.0f / eta : eta;
            Vector3float normal = outside ? n : -1.0f * n;
            float F0 = (eta - 1.0f) / (eta + 1.0f);
            F0 *= F0;

            float p_reflect = Math::fresnel_schlick(F0, fabsf(Math::dot(inc_dir, normal)));
            float xi = Math::rnd(seed);

            Vector3float refraction_dir = Math::refract(_eta, inc_dir, normal);
            Vector3float direction;
            if (xi <= p_reflect || Math::safeFloatEqual(Math::norm(refraction_dir), 0.0f))
            {
                direction = Math::reflect(inc_dir, normal);
            }
            else
            {
                direction = Math::normalize(refraction_dir);
            }

            return Vector4float(direction, 1);
        }

        private:
        Vector3float albedo_e;
        Vector3float albedo_d;
        Vector3float albedo_s;
        float eta;
    };

    DEFINE_PLUGIN(MaterialGlass, "GLASS", "1.0", Material)

}