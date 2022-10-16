#ifndef __CUPBR_SCENE_SCENE_H
#define __CUPBR_SCENE_SCENE_H

#include <Scene/SDF.h>
#include <Geometry/Geometry.h>
#include <DataStructure/Light.h>
#include <DataStructure/Image.h>
#include <DataStructure/Volume.h>
#include <DataStructure/BoundingVolumeHierarchy.h>

namespace cupbr
{
    /**
    *   @brief Struct to model a scene
    */
    struct Scene
    {
        virtual ~Scene() = default;

        std::vector<Properties> properties;         /**< Host vector of properties for materials */
    };

    struct GeometryScene : public Scene
    {
        Geometry** geometry = nullptr;              /**< The scene geometry */
        uint32_t scene_size = 0;                    /**< The number of objects in the scene */
        Light** lights = nullptr;                   /**< The light sources in the scene */
        uint32_t light_count = 0;                   /**< The light source count */
        bool useEnvironmentMap = false;             /**< Wether an environment map is loaded */
        Image<Vector3float> environment = {};       /**< The Environment map */
        Volume volume = {};                         /**< The volume inside the scene */

        /**
        *   @brief Get a scene element
        *   @param[in] index The index
        *   @return The geometry
        */
        CUPBR_HOST_DEVICE
        Geometry* operator[](const uint32_t index) const;

        BoundingVolumeHierarchy* bvh = nullptr;    /**< The Bounding volume hierarchy used for intersection testing */
    };

    struct SDFScene : public Scene
    {
        SDF* sdf = nullptr;                         /**< Models the csg */
    };
} //namespace cupbr

#include "../../src/Scene/SceneDetail.h"

#endif