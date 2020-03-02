#pragma once
#include "../Rendering/DrawCommand.h"
#include "FluidConfig.cuh"
class Fluid {
public:
	virtual void simulationStep() = 0;
	virtual void draw(const DrawCommand& drawCommand) = 0;
	virtual void init(FluidConfig config) = 0;
	virtual ~Fluid() {};

	float physicalTime;  // Time elapsed in the simulation.
};
