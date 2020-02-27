#pragma once

#include "Fluid.h"
#include "../Common/GpuCommons.h"


class Fluid_3D : public Fluid {
public:
	virtual glm::vec2 getCenter() = 0;
};