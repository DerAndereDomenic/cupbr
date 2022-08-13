#include <CUPBR.h>

namespace cupbr
{
    
    class MaterialMirror : public Material
    {
        public:

        MaterialMirror(const Properties& properties)
        {
            type = MaterialType::OPAQUE;
            albedo_e = properties.getProperty("albedo_e", Vector3float(0));
            albedo_s = properties.getProperty("albedo_s", Vector3float(0));
        }

        __host__ __device__
        virtual Vector3float
        MaterialMirror::brdf(const Vector3float& position, const Vector3float& inc_dir, const Vector3float& out_dir, const Vector3float& normal)
        {
            Vector3float reflected = Math::reflect(inc_dir, normal);
            return albedo_s * Math::delta(1.0f - Math::dot(out_dir, reflected)) / Math::dot(out_dir, normal);
        }

        __host__ __device__
        virtual Vector4float
        MaterialMirror::sampleDirection(uint32_t& seed, const Vector3float& inc_dir, const Vector3float& normal)
        {
            Vector3float direction = Math::reflect(inc_dir, normal);

            return Vector4float(direction, 1.0f);
        }

        private:
        Vector3float albedo_e;
        Vector3float albedo_s;
    };

    DEFINE_PLUGIN(MaterialMirror, "MIRROR", "1.0")

}