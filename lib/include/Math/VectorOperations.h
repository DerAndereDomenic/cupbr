#ifndef __CUPBR_MATH_VECTOROPERATIONS_H
#define __CUPBR_MATH_VECTOROPERATIONS_H

#include <Core/CUDA.cuh>
#include "VectorTypes_fwd.h"

template<typename T>
__host__ __device__
Vector2<T> operator+(const Vector2<T>& lhs, const Vector2<T>& rhs);

template<typename T>
__host__ __device__
Vector2<T> operator-(const Vector2<T>& lhs, const Vector2<T>& rhs);
 
template<typename T>
__host__ __device__
Vector2<T> operator*(const Vector2<T>& lhs, const T& rhs);

template<typename T>
__host__ __device__
Vector2<T> operator*(const T& lhs, const Vector2<T>& rhs);

template<typename T>
__host__ __device__
Vector2<T> operator/(const Vector2<T>& lhs, const T& rhs);

template<typename T>
__host__ __device__
void operator+=(Vector2<T>& lhs, const Vector2<T>& rhs);

template<typename T>
__host__ __device__
void operator-=(Vector2<T>& lhs, const Vector2<T>& rhs);
 
template<typename T>
__host__ __device__
void operator*=(Vector2<T> & lhs, const T & rhs);

template<typename T>
__host__ __device__
void operator/=(Vector2<T>& lhs, const T& rhs);

template<typename T>
__host__ __device__
Vector3<T> operator+(const Vector3<T>& lhs, const Vector3<T>& rhs);

template<typename T>
__host__ __device__
Vector3<T> operator-(const Vector3<T>& lhs, const Vector3<T>& rhs);

template<typename T>
__host__ __device__
Vector3<T> operator*(const Vector3<T>& lhs, const T& rhs);

template<typename T>
__host__ __device__
Vector3<T> operator*(const T& lhs, const Vector3<T>& rhs);

template<typename T>
__host__ __device__
Vector3<T> operator/(const Vector3<T>& lhs, const T& rhs);

template<typename T>
__host__ __device__
void operator+=(Vector3<T>& lhs, const Vector3<T>& rhs);

template<typename T>
__host__ __device__
void operator-=(Vector3<T>& lhs, const Vector3<T>& rhs);

template<typename T>
__host__ __device__
void operator*=(Vector3<T>& lhs, const T& rhs);

template<typename T>
__host__ __device__
void operator/=(Vector3<T>& lhs, const T& rhs);

#include "../../src/Math/VectorOperationsDetail.h"

#endif