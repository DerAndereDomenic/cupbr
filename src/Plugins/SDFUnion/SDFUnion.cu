#include <CUPBR.h>

namespace cupbr
{
    
    class SDFUnion : public SDF
    {
        public:

        SDFUnion(Properties* properties)
        {
            lhs = static_cast<SDF*>(properties->getProperty("lhs", (void*)nullptr));
            rhs = static_cast<SDF*>(properties->getProperty("rhs", (void*)nullptr));
        }

        CUPBR_HOST_DEVICE
        virtual float operator()(const Vector3float& x) 
        { 
            return fminf((*lhs)(x), (*rhs)(x));
        }

        private:
        SDF* lhs;
        SDF* rhs;
    };

    DEFINE_PLUGIN(SDFUnion, "SDFUnion", "1.0", SDF)

}