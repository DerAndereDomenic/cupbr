#ifndef __CUPBR_CUPBR_H
#define __CUPBR_CUPBR_H

/// ------- GL ---------

#include <GL/GLRenderer.h>
#include <GL/Window.h>

/// ------- CORE ---------

#include <Core/Platform.h>
#include <Core/CUPBRAPI.h>
#include <Core/CUDA.h>
#include <Core/Plugin.h>
#include <Core/KernelHelper.h>
#include <Core/Memory.h>
#include <Core/Tracing.h>
#include <Core/Event.h>
#include <Core/KeyEvent.h>
#include <Core/MouseEvent.h>
#include <Core/Properties.h>

/// ------- DATASTRUCTURE ---------

#include <DataStructure/Camera.h>
#include <DataStructure/Image.h>
#include <DataStructure/Light.h>
#include <DataStructure/RenderBuffer.h>

/// ------- GEOMETRY ---------

#include <Geometry/Geometry.h>
#include <Geometry/Material.h>
#include <Geometry/Mesh.h>
#include <Geometry/Plane.h>
#include <Geometry/Quad.h>
#include <Geometry/Ray.h>
#include <Geometry/Sphere.h>
#include <Geometry/Triangle.h>


/// ------- Interaction ---------

#include <Interaction/Interactor.h>
#include <Interaction/MousePicker.h>

/// ------- Math ---------

#include <Math/Functions.h>
#include <Math/VectorTypes_fwd.h>
#include <Math/Vector.h>
#include <Math/VectorTypes.h>
#include <Math/VectorFunctions.h>
#include <Math/VectorOperations.h>
#include <Math/MatrixTypes_fwd.h>
#include <Math/Matrix.h>
#include <Math/MatrixTypes.h>
#include <Math/MatrixFunctions.h>
#include <Math/MatrixOperations.h>

/// ------- Renderer ---------

#include <Renderer/PBRenderer.h>
#include <Renderer/ToneMapper.h>
#include <Renderer/RenderMethod.h>

/// ------- Scene ---------

#include <Scene/ObjLoader.h>
#include <Scene/Scene.h>
#include <Scene/SceneLoader.h>

#endif