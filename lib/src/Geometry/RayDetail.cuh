#ifndef __CUPBR_GEOMETRY_RAYDETAIL_CUH
#define __CUPBR_GEOMETRY_RAYDETAIL_CUH


Ray::Ray(const Vector3float& origin, const Vector3float direction)
    :_origin(origin),
     _direction(direction)
{
    Math::normalize(_direction);
}

__host__ __device__
Vector3float&
Ray::origin()
{
    return _origin;
}

__host__ __device__
Vector3float&
Ray::direction()
{
    return _direction;
}


#endif