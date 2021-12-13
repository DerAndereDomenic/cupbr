#include <DataStructure/BoundingVolumeHierarchy.h>

#include <algorithm>
#include <numeric>
#include <Core/Memory.h>

namespace cupbr
{
    namespace detail
    {
        inline bool box_compare(const Geometry* geom1, const Geometry* geom2, const uint32_t& axis)
        {
            return geom1->aabb().minimum()[axis] < geom2->aabb().minimum()[axis];
        }

        inline bool box_compare_x(const Geometry* geom1, const Geometry* geom2)
        {
            return box_compare(geom1, geom2, 0);
        }

        inline bool box_compare_y(const Geometry* geom1, const Geometry* geom2)
        {
            return box_compare(geom1, geom2, 1);
        }

        inline bool box_compare_z(const Geometry* geom1, const Geometry* geom2)
        {
            return box_compare(geom1, geom2, 2);
        }

        inline AABB surrounding_box(const AABB& box0, const AABB& box1)
        {
            Vector3float small(fminf(box0.minimum().x, box1.minimum().x),
                 fminf(box0.minimum().y, box1.minimum().y),
                 fminf(box0.minimum().z, box1.minimum().z));

            Vector3float big(fmaxf(box0.maximum().x, box1.maximum().x),
               fmaxf(box0.maximum().y, box1.maximum().y),
               fmaxf(box0.maximum().z, box1.maximum().z));

            return AABB(small,big);
        }
    
    } //namespace detail

    BoundingVolumeHierarchy::BoundingVolumeHierarchy(std::vector<Geometry*> host_objects,
                                                     std::vector<Geometry*> dev_objects)
    {
        type = GeometryType::BVH;
        uint32_t axis = (rand() / RAND_MAX) % 3;
        auto comparator = 
            (axis == 0) ? detail::box_compare_x
            : (axis == 1) ? detail::box_compare_y
            : detail::box_compare_z;

        size_t object_span = host_objects.size();

        Geometry* host_left, *host_right;
        BoundingVolumeHierarchy temp_left, temp_right;

        if(object_span == 1)
        {
            _left = _right = dev_objects[0];
            host_left = host_right = host_objects[0];
        }
        else if(object_span == 2)
        {
            if(comparator(host_objects[0], host_objects[1]))
            {
                //For the leaves -> store device objects
                //For further calculations -> store host
                _left = dev_objects[0];
                _right = dev_objects[1];
                host_left = host_objects[0];
                host_right = host_objects[1];
            }
            else
            {
                //For the leaves -> store device objects
                //For further calculations -> store host
                _left = dev_objects[1];
                _right = dev_objects[0];
                host_left = host_objects[1];
                host_right = host_objects[0];
            }
        }
        else
        {
            std::vector<size_t> idx(object_span);
            std::iota(idx.begin(), idx.end(), 0);

            std::stable_sort(idx.begin(), idx.end(),
                             [&host_objects,comparator](size_t i1, size_t i2)
            {
                return comparator(host_objects[i1], host_objects[i2]);
            });

            std::vector<Geometry*> host_left_objects;
            std::vector<Geometry*> host_right_objects;
            std::vector<Geometry*> dev_left_objects;
            std::vector<Geometry*> dev_right_objects;

            size_t mid = object_span / 2;
            for(uint32_t i = 0; i < object_span; ++i)
            {
                if(i < mid)
                {
                    host_left_objects.push_back(host_objects[idx[i]]);
                    dev_left_objects.push_back(dev_objects[idx[i]]);
                }
                else
                {
                    host_right_objects.push_back(host_objects[idx[i]]);
                    dev_right_objects.push_back(dev_objects[idx[i]]);
                }
            }

            temp_left = BoundingVolumeHierarchy(host_left_objects, dev_left_objects);
            temp_right = BoundingVolumeHierarchy(host_right_objects, dev_right_objects);
            host_left = &temp_left;
            host_right = &temp_right;

            _left = Memory::createDeviceObject<BoundingVolumeHierarchy>();
            _right = Memory::createDeviceObject<BoundingVolumeHierarchy>();

            Memory::copyHost2DeviceObject<BoundingVolumeHierarchy>(static_cast<BoundingVolumeHierarchy*>(host_left), static_cast<BoundingVolumeHierarchy*>(_left));
            Memory::copyHost2DeviceObject<BoundingVolumeHierarchy>(static_cast<BoundingVolumeHierarchy*>(host_right), static_cast<BoundingVolumeHierarchy*>(_right));
        
        }

        _type_left = host_left->type;
        _type_right = host_right->type;

        //Has to be done on host
        _aabb = detail::surrounding_box(host_left->aabb(), host_right->aabb());
    }

    void
    BoundingVolumeHierarchy::destroy()
    {
        if (_type_left == GeometryType::BVH)
        {
            BoundingVolumeHierarchy b;
            Memory::copyDevice2HostObject<BoundingVolumeHierarchy>(static_cast<BoundingVolumeHierarchy*>(_left), &b);
            b.destroy();
            Memory::destroyDeviceObject<BoundingVolumeHierarchy>(static_cast<BoundingVolumeHierarchy*>(_left));
        }

        if (_type_right == GeometryType::BVH)
        {
            BoundingVolumeHierarchy b;
            Memory::copyDevice2HostObject<BoundingVolumeHierarchy>(static_cast<BoundingVolumeHierarchy*>(_right), &b);
            b.destroy();
            Memory::destroyDeviceObject<BoundingVolumeHierarchy>(static_cast<BoundingVolumeHierarchy*>(_right));
        }
    }

} //namespace cupbr