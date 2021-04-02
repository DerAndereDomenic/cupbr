#ifndef __CUPBR_MATH_VECTORTYPESDETAIL_H
#define __CUPBR_MATH_VECTORTYPESDETAIL_H

template<typename T>
Vector2<T>::Vector2(const T& x, const T& y)
	:x(x),
	 y(y)
{

}

template<typename T>
Vector3<T>::Vector3(const T& x, const T& y, const T& z)
	:x(x),
	 y(y),
	 z(z)
{

}

template<typename T>
Vector4<T>::Vector4(const T& x, const T& y, const T& z, const T& w)
	:x(x),
	 y(y),
	 z(z),
	 w(w)
{

}

#endif