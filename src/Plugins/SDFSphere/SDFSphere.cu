#include <CUPBR.h>

namespace cupbr
{
    
    class SDFSphere : public SDF
    {
        public:

        SDFSphere(Properties* properties)
        {
            pos = properties->getProperty("position", Vector3float(0));
            radius = properties->getProperty("radius", 1.0f);
        }

        CUPBR_HOST_DEVICE
        virtual float operator()(const Vector3float& x) 
        { 
            return Math::dot(x, pos) - radius;
        }

        private:
        Vector3float pos;
        float radius;
    };

    DEFINE_PLUGIN(SDFSphere, "SDFSphere", "1.0", SDF)

}