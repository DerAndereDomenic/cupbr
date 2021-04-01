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
void 
Math::normalize(Vector2<T>& v)
{
	v /= norm(v);
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
void 
Math::normalize(Vector3<T>& v)
{
	v /= norm(v);
}

#endif