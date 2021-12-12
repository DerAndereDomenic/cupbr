#include <Geometry/Geometry.h>


namespace cupbr
{
    static uint32_t current_geometry_id = 0;

    Geometry::Geometry()
        :_id(current_geometry_id++)
    {

    }
}