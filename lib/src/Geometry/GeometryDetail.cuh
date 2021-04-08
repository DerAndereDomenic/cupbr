#ifndef __CUPBR_GEOMETRY_GEOMETRYDETAIL_CUH
#define __CUPBR_GEOMETRY_GEOMETRYDETAIL_CUH

__host__ __device__
Vector4float
Geometry::computeRayIntersection(const Ray& ray)
{
    return Vector4float(INFINITY);
}

__host__ __device__
Vector3float
Geometry::getNormal(const Vector3float& x)
{
    return Vector3float(INFINITY);
}

#endif