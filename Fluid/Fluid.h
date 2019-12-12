#pragma once
#include "../Rendering/DrawCommand.h"
#include "FluidConfig.cuh"
class Fluid {
	virtual void simulationStep() = 0;
	virtual void draw(const DrawCommand& drawCommand) = 0;
	virtual void init(std::shared_ptr<FluidConfig> config) = 0;
};
