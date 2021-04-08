#ifndef __CUPBR_MATH_VECTOROPERATIONSDETAIL_H
#define __CUPBR_MATH_VECTOROPERATIONSDETAIL_H

#include <Math/VectorOperations.h>

template<typename T>
__host__ __device__
Vector2<T> operator+(const Vector2<T>& lhs, const Vector2<T>& rhs)
{
	return Vector2<T>(lhs.x + rhs.x, lhs.y + rhs.y);
}

template<typename T>
__host__ __device__
Vector2<T> operator-(const Vector2<T>& lhs, const Vector2<T>& rhs)
{
	return Vector2<T>(lhs.x - rhs.x, lhs.y - rhs.y);
}

template<typename T>
__host__ __device__
Vector2<T> operator*(const Vector2<T>& lhs, const T& rhs)
{
	return operator*(rhs, lhs);
}

template<typename T>
__host__ __device__
Vector2<T> operator*(const T& lhs, const Vector2<T>& rhs)
{
	return Vector2<T>(lhs * rhs.x, lhs * rhs.y);
}

template<typename T>
__host__ __device__
Vector2<T> operator*(const Vector2<T>& lhs, const Vector2<T>& rhs)
{
	return Vector2<T>(lhs.x*rhs.x, lhs.y*rhs.y);
}

template<typename T>
__host__ __device__
Vector2<T> operator/(const Vector2<T>& lhs, const T& rhs)
{
	return Vector2<T>(lhs.x / rhs, lhs.y / rhs);
}

template<typename T>
__host__ __device__
void operator+=(Vector2<T>& lhs, const Vector2<T>& rhs)
{
	lhs.x += rhs.x;
	lhs.y += rhs.y;
}

template<typename T>
__host__ __device__
void operator-=(Vector2<T>& lhs, const Vector2<T>& rhs)
{
	lhs.x -= rhs.x;
	lhs.y -= rhs.y;
}

template<typename T>
__host__ __device__
void operator*=(Vector2<T>& lhs, const T& rhs)
{
	lhs.x *= rhs;
	lhs.y *= rhs;
}

template<typename T>
__host__ __device__
void operator/=(Vector2<T>& lhs, const T& rhs)
{
	lhs.x /= rhs;
	lhs.y /= rhs;
}

template<typename T>
__host__ __device__
Vector3<T> operator+(const Vector3<T>& lhs, const Vector3<T>& rhs)
{
	return Vector3<T>(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
}

template<typename T>
__host__ __device__
Vector3<T> operator-(const Vector3<T>& lhs, const Vector3<T>& rhs)
{
	return Vector3<T>(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}

template<typename T>
__host__ __device__
Vector3<T> operator*(const Vector3<T>& lhs, const T& rhs)
{
	return operator*(rhs, lhs);
}

template<typename T>
__host__ __device__
Vector3<T> operator*(const T& lhs, const Vector3<T>& rhs)
{
	return Vector3<T>(lhs * rhs.x, lhs * rhs.y, lhs * rhs.z);
}

template<typename T>
__host__ __device__
Vector3<T> operator*(const Vector3<T>& lhs, const Vector3<T>& rhs)
{
	return Vector3<T>(lhs.x*rhs.x, lhs.y*rhs.y, lhs.z*rhs.z);
}

template<typename T>
__host__ __device__
Vector3<T> operator/(const Vector3<T>& lhs, const T& rhs)
{
	return Vector3<T>(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs);
}

template<typename T>
__host__ __device__
void operator+=(Vector3<T>& lhs, const Vector3<T>& rhs)
{
	lhs.x += rhs.x;
	lhs.y += rhs.y;
	lhs.z += rhs.z;
}

template<typename T>
__host__ __device__
void operator-=(Vector3<T>& lhs, const Vector3<T>& rhs)
{
	lhs.x -= rhs.x;
	lhs.y -= rhs.y;
	lhs.z -= rhs.z;
}

template<typename T>
__host__ __device__
void operator*=(Vector3<T>& lhs, const T& rhs)
{
	lhs.x *= rhs;
	lhs.y *= rhs;
	lhs.z *= rhs;
}

template<typename T>
__host__ __device__
void operator/=(Vector3<T>& lhs, const T& rhs)
{
	lhs.x /= rhs;
	lhs.y /= rhs;
	lhs.z /= rhs;
}

template<typename T>
__host__ __device__
Vector4<T> operator+(const Vector4<T>& lhs, const Vector4<T>& rhs)
{
	return Vector4<T>(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w);
}

template<typename T>
__host__ __device__
Vector4<T> operator-(const Vector4<T>& lhs, const Vector4<T>& rhs)
{
	return Vector3<T>(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w);
}

template<typename T>
__host__ __device__
Vector4<T> operator*(const Vector4<T>& lhs, const T& rhs)
{
	return operator*(rhs, lhs);
}

template<typename T>
__host__ __device__
Vector4<T> operator*(const T& lhs, const Vector4<T>& rhs)
{
	return Vector4<T>(lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, lhs * rhs.w);
}

template<typename T>
__host__ __device__
Vector4<T> operator*(const Vector4<T>& lhs, const Vector4<T>& rhs)
{
	return Vector4<T>(lhs.x*rhs.x, lhs.y*rhs.y, lhs.z*rhs.z, lhs.w*rhs.w);
}

template<typename T>
__host__ __device__
Vector4<T> operator/(const Vector4<T>& lhs, const T& rhs)
{
	return Vector4<T>(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs);
}

template<typename T>
__host__ __device__
void operator+=(Vector4<T>& lhs, const Vector4<T>& rhs)
{
	lhs.x += rhs.x;
	lhs.y += rhs.y;
	lhs.z += rhs.z;
	lhs.w += rhs.w;
}

template<typename T>
__host__ __device__
void operator-=(Vector4<T>& lhs, const Vector4<T>& rhs)
{
	lhs.x -= rhs.x;
	lhs.y -= rhs.y;
	lhs.z -= rhs.z;
	lhs.w -= rhs.w;
}

template<typename T>
__host__ __device__
void operator*=(Vector4<T>& lhs, const T& rhs)
{
	lhs.x *= rhs;
	lhs.y *= rhs;
	lhs.z *= rhs;
	lhs.w *= rhs;
}

template<typename T>
__host__ __device__
void operator/=(Vector4<T>& lhs, const T& rhs)
{
	lhs.x /= rhs;
	lhs.y /= rhs;
	lhs.z /= rhs;
	lhs.w /= rhs;
}

#endif