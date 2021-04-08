#ifndef __CUPBR_GEOMETRY_SPHEREDETAIL_CUH
#define __CUPBR_GEOMETRY_SPHEREDETAIL_CUH

__host__ __device__
inline
Sphere::Sphere(const Vector3float& position, const float& radius)
    :_position(position),
     _radius(radius)
{

}

__host__ __device__
inline Vector3float
Sphere::position() const
{
    return _position;
}

__host__ __device__
inline float
Sphere::radius() const
{
    return _radius;
}

__host__ __device__
inline Vector4float
Sphere::computeRayIntersection(const Ray& ray)
{
    const Vector3float origin = ray.origin();
    const Vector3float direction = ray.direction();

    Vector3float OS = origin - _position;
    float b = 2.0f * Math::dot(direction, OS);
    float c = Math::dot(OS, OS) - _radius * _radius;
    float disc = b * b - 4 * c;

    if(disc > 0)
    {
        float distSqrt = sqrtf(disc);
        float q = b < 0 ? (-b - distSqrt) / 2.0f : (-b + distSqrt) / 2.0f;
        float t0 = q;
        float t1 = c/q;

        t0 = min(t0,t1);
        t1 = max(t0,t1);

        if(t1 >= 0)
        {
            float t = t0 < 0 ? t1 : t0;
            return Vector4float(origin + t*direction,t);
        }
    }
    return Vector4float(INFINITY);
}

__host__ __device__
inline Vector3float
Sphere::getNormal(const Vector3float& x)
{
    return Math::normalize(x-_position);
}

#endif