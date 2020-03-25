#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <string>

#include <memory>
#include "../Common/GpuCommons.h"

// Values are default values


enum class ShapeType {
	Sphere,Square
};

struct InitializationVolume {

	ShapeType shapeType = ShapeType::Square;

	float3 boxMin = make_float3(0,0,0);
	float3 boxMax = make_float3(1, 0.1875, 1);

	float3 ballCenter = make_float3(0.5, 0.75, 0.5);
	float ballRadius = 0.125;

	int phase = 0;

	InitializationVolume() {

	}

	InitializationVolume(ShapeType type) {
		shapeType = type;
	}
};

struct ConfigFLIP {
	int sizeX = 50;
	int sizeY = 50;
	int sizeZ = 50;
	int pressureIterations = 100;
	int diffusionIterations = 100;
	float timestep = 0.033;
	float FLIPcoeff = 0.95;
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
		initialVolumes.emplace_back(ShapeType::Square);
		
		initialVolumes.emplace_back(ShapeType::Sphere);

		phaseColors.push_back(make_float4(0, 0, 1, 0.2));
		phaseColors.push_back(make_float4(1, 0, 0, 1));
	}
};



//FluidConfig getConfigFromFile();

