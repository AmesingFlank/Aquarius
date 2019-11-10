#pragma once
#include "../Rendering/DrawCommand.h"

class Fluid {
	virtual void simulationStep() = 0;
	virtual void draw(const DrawCommand& drawCommand) = 0;
};
