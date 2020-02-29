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

// Values are default values
struct ConfigFLIP {
	int sizeX = 50;
	int sizeY = 50;
	int sizeZ = 50;
	int pressureIterations = 100;
	int diffusionIterations = 50;
};

struct ConfigPBF {
	int substeps = 4;
	int iterations = 4;
	int maxParticleCount = 3e5;
};

struct ConfigPCISPH {
	int substeps = 1;
	int iterations = 4;
	int maxParticleCount = 3e5;
	float stiffness = 15;
};

struct FluidConfig{



	// Simulation Set-up
	std::string method = "FLIP";
	std::vector<InitializationVolume> initialVolumes;
	float timestep = 0.033;
	float gravity = -9.8;


	// Multiphase Settings
	int phaseCount = 2;
	std::vector<float4> phaseColors;
	float diffusionCoeff = 0.01;


	ConfigFLIP FLIP;
	ConfigPBF PBF;
	ConfigPCISPH PCISPH;

	FluidConfig() {
		initialVolumes.push_back(
			{
				ShapeType::Square,
				std::vector<float>({0,0,0, 0.5,0.2,1}),
				0
			}
		);
		initialVolumes.push_back(
			{
				ShapeType::Square,
				std::vector<float>({0.5,0,0, 1.0,0.2,1}),
				1
			}
		);
		initialVolumes.push_back(
			{
				ShapeType::Sphere,
				std::vector<float>({0.5,0.8,0.5,   0.15}),
				1
			}
		);

		phaseColors.push_back(make_float4(0, 0, 1, 1));
		phaseColors.push_back(make_float4(1, 0, 1, 1));
	}
};



//FluidConfig getConfigFromFile();

