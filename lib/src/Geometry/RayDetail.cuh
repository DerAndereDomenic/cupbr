#ifndef __CUPBR_GEOMETRY_RAYDETAIL_CUH
#define __CUPBR_GEOMETRY_RAYDETAIL_CUH

__host__ __device__
Ray::Ray(const Vector3float& origin, const Vector3float direction)
    :_origin(origin),
     _direction(Math::normalize(direction))
{

}

__host__ __device__
Vector3float
Ray::origin() const
{
    return _origin;
}

__host__ __device__
Vector3float
Ray::direction() const
{
    return _direction;
}


#endif