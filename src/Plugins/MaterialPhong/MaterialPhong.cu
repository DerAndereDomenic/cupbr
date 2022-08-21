#include <CUPBR.h>

namespace cupbr
{
    
    class MaterialPhong : public Material
    {
        public:

        MaterialPhong(Properties* properties)
        {
            type = MaterialType::OPAQUE;
            albedo_e = properties->getProperty("albedo_e", Vector3float(0));
            albedo_d = properties->getProperty("albedo_d", Vector3float(1));
            albedo_s = properties->getProperty("albedo_s", Vector3float(0));
            shininess = properties->getProperty("shininess", 128.0f * 0.4f);
        }

        CUPBR_HOST_DEVICE
        virtual Vector3float
        MaterialPhong::brdf(const Vector3float& position, const Vector3float& inc_dir, const Vector3float& out_dir, const Vector3float& normal)
        {
            Vector3float halfDir = Math::normalize(inc_dir + out_dir);
            return albedo_s * powf(fmaxf(0.0f, Math::dot(halfDir, normal)), shininess) + albedo_d/static_cast<float>(M_PI);
        }

        CUPBR_HOST_DEVICE
        virtual Vector4float
        MaterialPhong::sampleDirection(uint32_t& seed, const Vector3float& inc_dir, const Vector3float& normal)
        {
            const float xi_1 = Math::rnd(seed);
            const float xi_2 = Math::rnd(seed);

            const float r = sqrtf(xi_1);
            const float phi = 2.0f * 3.14159f * xi_2;

            const float x = r * cos(phi);
            const float y = r * sin(phi);
            const float z = sqrtf(fmaxf(0.0f, 1.0f - x * x - y * y));

            Vector3float direction = Math::normalize(Math::toLocalFrame(normal, Vector3float(x, y, z)));

            float p = fmaxf(EPSILON, Math::dot(direction, normal)) / 3.14159f;

            return Vector4float(direction, p);
        }

        private:
        Vector3float albedo_e;
        Vector3float albedo_d;
        Vector3float albedo_s;
        float shininess;
    };

    DEFINE_PLUGIN(MaterialPhong, "PHONG", "1.0", Material)

}