#ifndef __CUPBR_RENDERER_TONEMAPPINGMETHOD_H
#define __CUPBR_RENDERER_TONEMAPPINGMETHOD_H

#include <Core/Plugin.h>

namespace cupbr
{
    class ToneMappingMethod : public Plugin
    {
        public:
        ToneMappingMethod() = default;

        ToneMappingMethod(Properties* properties) {}

        virtual void toneMap(Image<Vector3float>& hdr_image,
                             RenderBuffer& output_img) {}
    };

} //namespace cupbr

#endif
