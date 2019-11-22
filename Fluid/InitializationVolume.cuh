#pragma once

#include <vector>
#include "../GpuCommons.h"

enum class ShapeType {
	Sphere,Square
};

struct InitializationVolumn2D {

	ShapeType shapeType;

	std::vector< float2 > params;

};

struct InitializationVolumn3D {

	ShapeType shapeType;

	std::vector< float3 > params;

};