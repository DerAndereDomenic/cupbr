#ifndef __CUPBR_CORE_TRACINGDETAIL_CUH
#define __CUPBR_CORE_TRACINGDETAIL_CUH

__device__
Ray
Tracing::launchRay(const uint32_t& tid, const uint32_t& width, const uint32_t& height, const Camera& camera)
{
    const Vector2uint32_t pixel = ThreadHelper::index2pixel(tid, width, height);

    const float ratio_x = 2.0f*(static_cast<float>(pixel.x)/width - 0.5f);
    const float ratio_y = 2.0f*(static_cast<float>(pixel.y)/height - 0.5f);

    const Vector3float world_pos = camera.position() + camera.zAxis() + ratio_x*camera.xAxis() + ratio_y*camera.yAxis();

    return Ray(camera.position(), world_pos - camera.position());
}

#endif