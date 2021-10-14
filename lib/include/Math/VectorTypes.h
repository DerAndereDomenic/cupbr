#ifndef __CUPBR_MATH_VECTORTYPES_H
#define __CUPBR_MATH_VECTORTYPES_H

#include <Math/VectorTypes_fwd.h>
#include <Core/CUDA.cuh>

namespace cupbr
{
    /**
    *	@brief A 2D vector
    */
    template<typename T>
    struct Vector2
    {
        public:
        /**
        *	@brief Default constructor.
        */
        Vector2() = default;

        /**
        *	@brief Creates a 2D vector.
        *	@param[in] x The x coordinate
        *	@param[in] y The y coordinate
        */
        __host__ __device__
        Vector2(const T& x, const T& y);

        /**
        *	@brief Creats a constant 2D vector.
        *	@param[in] v The value
        */
        __host__ __device__
        Vector2(const T& v);

        /**
        *	@brief Cast between vector types.
        */
        template<typename S>
        __host__ __device__
        operator Vector2<S>() const
        {
            return Vector2<S>(static_cast<S>(x), static_cast<S>(y));
        }

        T x;	/**< The x coordinate*/
        T y;	/**< The y coordinate*/
    };

    /**
    *	@brief A 3D vector
    */
    template<typename T>
    struct Vector3
    {
        public:
        /**
        *	@brief Default constructor.
        */
        Vector3() = default;

        /**
        *	@brief Creates a 3D vector.
        *	@param[in] x The x coordinate
        *	@param[in] y The y coordinate
        *	@param[in] z The z coordinate
        */
        __host__ __device__
        Vector3(const T& x, const T& y, const T& z);

        /**
        *	@brief Creats a constant 3D vector.
        *	@param[in] v The value
        */
        __host__ __device__
        Vector3(const T& v);

        /**
        *	@brief Creates a 3D vector from a 4D vector
        *	@param[in] v The 4D vector
        *	@note The w component is discarded
        */
        __host__ __device__
        Vector3(const Vector4<T>& v);

        /**
        *	@brief Cast between vector types.
        */
        template<typename S>
        __host__ __device__
        operator Vector3<S>() const
        {
            return Vector3<S>(static_cast<S>(x), static_cast<S>(y), static_cast<S>(z));
        }

        T x;	/**< The x coordinate*/
        T y;	/**< The y coordinate*/
        T z;	/**< The z coordinate*/
    };

    /**
    *	@brief A 4D vector
    */
    template<typename T>
    struct Vector4
    {
        public:
        /**
        *	@brief Default constructor.
        */
        Vector4() = default;

        /**
        *	@brief Creates a 4D vector.
        *	@param[in] x The x coordinate
        *	@param[in] y The y coordinate
        *	@param[in] z The z coordinate
        *	@param[in] w The w coordinate
        */
        __host__ __device__
        Vector4(const T& x, const T& y, const T& z, const T& w);

        /**
        *	@brief Creates a 4D vector from a 3D one.
        *	@param[in] v A 3D vector
        *	@param[in] w The w component
        */
        __host__ __device__
        Vector4(const Vector3<T>& v, const T& w);

        /**
        *	@brief Creats a constant 4D vector.
        *	@param[in] v The value
        */
        __host__ __device__
        Vector4(const T& v);

        /**
        *	@brief Cast between vector types.
        */
        template<typename S>
        __host__ __device__
        operator Vector4<S>() const
        {
            return Vector4<S>(static_cast<S>(x), static_cast<S>(y), static_cast<S>(z), static_cast<S>(w));
        }

        T x;	/**< The x coordinate*/
        T y;	/**< The y coordinate*/
        T z;	/**< The z coordinate*/
        T w;	/**< The w coordinate*/
    };

} //namespace cupbr

#include "../../src/Math/VectorTypesDetail.h"

#endif