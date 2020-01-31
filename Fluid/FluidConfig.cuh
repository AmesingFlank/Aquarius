#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <string>
#include <rapidjson/reader.h>
#include <rapidjson/document.h>
#include <memory>
#include "../Common/GpuCommons.h"

enum class ShapeType {
	Sphere,Square
};

struct InitializationVolume {

	ShapeType shapeType;

	std::vector< float > params;

	int phase;
};


struct FluidConfig {
	int dimension;
	std::string method;
};

struct FluidConfig3D:public FluidConfig {
	int sizeX;
	int sizeY;
	int sizeZ;
	std::vector<InitializationVolume> initialVolumes;

	int phaseCount;
	std::vector<float4> phaseColors;

	float4* phaseColorsDevice;
};

struct FluidConfig2D:public FluidConfig {
	int sizeX;
	int sizeY;
	std::vector<InitializationVolume> initialVolumes;
};

std::shared_ptr<FluidConfig> getConfig();