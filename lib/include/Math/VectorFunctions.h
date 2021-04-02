#ifndef __CUPBR_MATH_VECTORFUNCTIONS_H
#define __CUPBR_MATH_VECTORFUNCTIONS_H

#include <Math/VectorTypes_fwd.h>
#include <Core/CUDA.cuh>

namespace Math
{
	/**
	*	@brief Get the norm of a vector
	*	@tparam T The vector data type
	*	@param v The vector
	*	@return The L2 norm
	*/
	template<typename T>
	__host__ __device__
	T norm(const Vector2<T>& v);

	/**
	*	@brief Calculates the dot product
	*	@tparam T The vector data type
	*	@param v The first vector
	*	@param w The second vector
	*	@return The dot product
	*/
	template<typename T>
	__host__ __device__
	T dot(const Vector2<T>& v, const Vector2<T>& w);

	/**
	*	@brief Normalizes the given vector
	*	@tparam T The vector data type
	*	@param v The vector to normalize
	*/
	template<typename T>
	__host__ __device__
	void normalize(Vector2<T>& v);

	/**
	*	@brief Get the norm of a vector
	*	@tparam T The vector data type
	*	@param v The vector
	*	@return The L2 norm
	*/
	template<typename T>
	__host__ __device__
	T norm(const Vector3<T>& v);

	/**
	*	@brief Calculates the dot product
	*	@tparam T The vector data type
	*	@param v The first vector
	*	@param w The second vector
	*	@return The dot product
	*/
	template<typename T>
	__host__ __device__
	T dot(const Vector3<T>& v, const Vector3<T>& w);

	/**
	*	@brief Normalizes the given vector
	*	@tparam T The vector data type
	*	@param v The vector to normalize
	*/
	template<typename T>
	__host__ __device__
	void normalize(Vector3<T>& v);
	
	/**
	*	@brief Get the norm of a vector
	*	@tparam T The vector data type
	*	@param v The vector
	*	@return The L2 norm
	*/
	template<typename T>
	__host__ __device__
	T norm(const Vector4<T>& v);

	/**
	*	@brief Calculates the dot product
	*	@tparam T The vector data type
	*	@param v The first vector
	*	@param w The second vector
	*	@return The dot product
	*/
	template<typename T>
	__host__ __device__
	T dot(const Vector4<T>& v, const Vector4<T>& w);

	/**
	*	@brief Normalizes the given vector
	*	@tparam T The vector data type
	*	@param v The vector to normalize
	*/
	template<typename T>
	__host__ __device__
	void normalize(Vector4<T>& v);
}

#include "../../src/Math/VectorFunctionsDetail.h"

#endif