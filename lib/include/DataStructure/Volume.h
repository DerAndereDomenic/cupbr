#ifndef __CUPBR_DATASTRUCTURE_VOLUME_H
#define __CUPBR_DATASTRUCTURE_VOLUME_H

namespace cupbr
{
    /**
    *   @brief A struct to model a volume
    */
    struct Volume
    {
        Vector3float sigma_s = 1.0f;    /**< The scattering coefficient */
        Vector3float sigma_a = 1.0f;    /**< The absorbtion coefficient */
        float g = 0.0f;                 /**< The phase function parameter */
    };
} //namespace cupbr

#endif