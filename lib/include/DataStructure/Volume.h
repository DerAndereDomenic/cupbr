#ifndef __CUPBR_DATASTRUCTURE_VOLUME_H
#define __CUPBR_DATASTRUCTURE_VOLUME_H

namespace cupbr
{
    enum class Interface
    {
        NONE = 0,
        GLASS = 1
    };

    /**
    *   @brief A struct to model a volume
    */
    struct Volume
    {
        Vector3float sigma_s = 0.0f;                        /**< The scattering coefficient */
        Vector3float sigma_a = 0.0f;                        /**< The absorbtion coefficient */
        float g = 0.0f;                                     /**< The phase function parameter */
        Interface interface = Interface::NONE;              /**< The volume interface */
    };
} //namespace cupbr

#endif