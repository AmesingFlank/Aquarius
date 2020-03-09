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

// Values are default values


enum class ShapeType {
	Sphere,Square
};

struct InitializationVolume {

	ShapeType shapeType = ShapeType::Square;

	std::vector< float > params = {0,0,0,0,0,0};

	int phase = 0;
};

struct ConfigFLIP {
	int sizeX = 50;
	int sizeY = 50;
	int sizeZ = 50;
	int pressureIterations = 100;
	int diffusionIterations = 100;
	float timestep = 0.033;
};

struct ConfigPBF {
	float timestep = 0.033;
	int substeps = 4;
	int iterations = 2;
	int maxParticleCount = 3e5;
};

struct ConfigPCISPH {
	float timestep = 0.005;
	int substeps = 1;
	int iterations = 4;
	int maxParticleCount = 3e5;
	float stiffness = 15;
};

struct FluidConfig{



	// Simulation Set-up
	std::string method = "FLIP";
	std::vector<InitializationVolume> initialVolumes;
	float3 gravity = make_float3(0,-9.8,0);


	// Multiphase Settings
	int phaseCount = 2;
	std::vector<float4> phaseColors;
	float4 phaseDensities = make_float4(1,1,1,1);
	float diffusionCoeff = 0.001;


	ConfigFLIP FLIP;
	ConfigPBF PBF;
	ConfigPCISPH PCISPH;

	FluidConfig() {
		initialVolumes.push_back(
			{
				ShapeType::Square,
				std::vector<float>({0,0,0, 1,0.2,1}),
				0
			}
		);
		
		initialVolumes.push_back(
			{
				ShapeType::Sphere,
				std::vector<float>({0.5,0.67,0.5,   0.27}),
				1
			}
		);

		phaseColors.push_back(make_float4(0, 0, 1, 0.2));
		phaseColors.push_back(make_float4(1, 0, 0, 1));
	}
};



//FluidConfig getConfigFromFile();

