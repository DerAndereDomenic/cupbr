#ifndef __CUPBR_GEOMETRY_RAYDETAIL_CUH
#define __CUPBR_GEOMETRY_RAYDETAIL_CUH

namespace cupbr
{
    __host__ __device__
    inline
    Ray::Ray(const Vector3float& origin, const Vector3float direction)
        :_origin(origin),
        _direction(Math::normalize(direction))
    {

    }

    __host__ __device__
    inline Vector3float
    Ray::origin() const
    {
        return _origin;
    }

    __host__ __device__
    inline Vector3float
    Ray::direction() const
    {
        return _direction;
    }

    __host__ __device__
    inline void
    Ray::traceNew(const Vector3float& origin, const Vector3float& direction)
    {
        _origin = origin;
        _direction = Math::normalize(direction);
    }

    template<class PayloadType>
    __host__ __device__
    void
    Ray::setPayload(PayloadType* payload)
    {
        _payload = static_cast<void*>(payload);
    }

    template<class PayloadType>
    __host__ __device__
    PayloadType*
    Ray::payload()
    {
        return static_cast<PayloadType*>(_payload);
    }
} //namespace cupbr

#endif