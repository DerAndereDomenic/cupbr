#ifndef __CUPBR_GEOMETRY_RAYDETAIL_H
#define __CUPBR_GEOMETRY_RAYDETAIL_H

namespace cupbr
{
    CUPBR_HOST_DEVICE
    inline
    Ray::Ray(const Vector3float& origin, const Vector3float direction)
        :_origin(origin),
        _direction(Math::normalize(direction))
    {

    }

    CUPBR_HOST_DEVICE
    inline Vector3float
    Ray::origin() const
    {
        return _origin;
    }

    CUPBR_HOST_DEVICE
    inline Vector3float
    Ray::direction() const
    {
        return _direction;
    }

    CUPBR_HOST_DEVICE
    inline void
    Ray::traceNew(const Vector3float& origin, const Vector3float& direction)
    {
        _origin = origin;
        _direction = Math::normalize(direction);
    }

    template<class PayloadType>
    CUPBR_HOST_DEVICE
    void
    Ray::setPayload(PayloadType* payload)
    {
        _payload = static_cast<void*>(payload);
    }

    template<class PayloadType>
    CUPBR_HOST_DEVICE
    PayloadType*
    Ray::payload()
    {
        return static_cast<PayloadType*>(_payload);
    }
} //namespace cupbr

#endif