#ifndef __CUPBR_CUPBR_H
#define __CUPBR_CUPBR_H

/// ------- GL ---------

#include <GL/GLRenderer.cuh>
#include <GL/Window.h>

/// ------- CORE ---------

#include <Core/CUDA.cuh>
#include <Core/KernelHelper.cuh>
#include <Core/Memory.cuh>
#include <Core/Tracing.cuh>

/// ------- DATASTRUCTURE ---------

#include <DataStructure/Camera.cuh>
#include <DataStructure/Image.cuh>
#include <DataStructure/Light.cuh>
#include <DataStructure/RenderBuffer.cuh>

/// ------- GEOMETRY ---------

#include <Geometry/Geometry.cuh>
#include <Geometry/Material.cuh>
#include <Geometry/Mesh.cuh>
#include <Geometry/Plane.cuh>
#include <Geometry/Quad.cuh>
#include <Geometry/Ray.cuh>
#include <Geometry/Sphere.cuh>
#include <Geometry/Triangle.cuh>


/// ------- Interaction ---------

#include <Interaction/Interactor.cuh>
#include <Interaction/MousePicker.cuh>

/// ------- Math ---------

#include <Math/Functions.cuh>
#include <Math/VectorTypes_fwd.h>
#include <Math/Vector.h>
#include <Math/VectorTypes.h>
#include <Math/VectorFunctions.h>
#include <Math/VectorOperations.h>

/// ------- PostProcessing ---------

#include <PostProcessing/Convolution.cuh>
#include <PostProcessing/PostProcessor.h>
#include <PostProcessing/PyramidConstructor.h>

/// ------- Renderer ---------

#include <Renderer/GradientDomain.cuh>
#include <Renderer/PathTracer.cuh>
#include <Renderer/PBRenderer.cuh>
#include <Renderer/RayTracer.cuh>
#include <Renderer/ToneMapper.cuh>
#include <Renderer/VolumeRenderer.cuh>
#include <Renderer/Whitted.cuh>

/// ------- Scene ---------

#include <Scene/ObjLoader.cuh>
#include <Scene/Scene.cuh>
#include <Scene/SceneLoader.cuh>

#endif