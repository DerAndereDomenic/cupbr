#include <CUPBR.h>

namespace cupbr
{
    
    class SDFCube : public SDF
    {
        public:

        SDFCube(Properties* properties)
        {
            pos = properties->getProperty("position", Vector3float(0));
            size = Vector3float(1,2,3);//properties->getProperty("size", Vector3float(1));
        }

        CUPBR_HOST_DEVICE
        virtual float operator()(const Vector3float& x) 
        { 
            float a = size.x;
            float b = size.y;
            float c = size.z;
            Vector3float offset = x - pos;
            Vector3float abs_vec = Vector3float(fabsf(offset.x) - a, fabsf(offset.y) - b, fabsf(offset.z) - c);

            return fmaxf(fmaxf(abs_vec.x, abs_vec.y), abs_vec.z);
        }

        private:
        Vector3float pos;
        Vector3float size;
    };

    DEFINE_PLUGIN(SDFCube, "SDFCube", "1.0", SDF)

}