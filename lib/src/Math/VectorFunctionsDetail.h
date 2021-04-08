#ifndef __CUPBR_MATH_VECTORFUNCTIONSDETAIL_H
#define __CUPBR_MATH_VECTORFUNCTIONSDETAIL_H

#include <Math/VectorTypes.h>
#include <Math/VectorOperations.h>

template<typename T>
T 
Math::norm(const Vector2<T>& v)
{
	return sqrtf(dot(v, v));
}

template<typename T>
T 
Math::dot(const Vector2<T>& v, const Vector2<T>& w)
{
	return v.x * w.x + v.y * w.y;
}

template<typename T>
Vector2<T> 
Math::normalize(const Vector2<T>& v)
{
	return v/norm(v);
}

template<typename T>
T 
Math::norm(const Vector3<T>& v)
{
	return sqrtf(dot(v, v));
}

template<typename T>
T 
Math::dot(const Vector3<T>& v, const Vector3<T>& w)
{
	return v.x * w.x + v.y * w.y + v.z * w.z;
}

template<typename T>
Vector3<T> 
Math::normalize(const Vector3<T>& v)
{
	return v/norm(v);
}

template<typename T>
T 
Math::norm(const Vector4<T>& v)
{
	return sqrtf(dot(v, v));
}

template<typename T>
T 
Math::dot(const Vector4<T>& v, const Vector4<T>& w)
{
	return v.x * w.x + v.y * w.y + v.z * w.z + v.w*w.w;
}

template<typename T>
Vector4<T> 
Math::normalize(const Vector4<T>& v)
{
	return v/norm(v);
}

#endif