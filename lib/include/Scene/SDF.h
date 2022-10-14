#ifndef __CUPBR_SCENE_SDF_H
#define __CUPBR_SCENE_SDF_H

#include <Core/Plugin.h>

namespace cupbr
{
    /**
    *   @brief Class to model an SDF
    */
    class SDF : public Plugin
    {
        public:
        SDF() = default;

        /**
        *   @brief Constructor
        *   @param properties The input properties
        */
        SDF(Properties& properties) {}

        /**
        *   @brief Evaluate the sdf
        *   @param x The position where the sdf should be evaluated
        *   @return The sdf value
        */
        CUPBR_HOST_DEVICE
        virtual float operator()(const Vector3float& x) { return 0; }
    };

    //min(a,b)
    class SDFUnion : public SDF
    {
        public:

        private:
    };

    //max(-a,b)
    class SDFDifference : public SDF
    {
        public:

        private:
    };

    //max(a,b)
    class SDFIntersection : public SDF
    {
        public:

        private:
    };


} //namespace cupbr

#endif