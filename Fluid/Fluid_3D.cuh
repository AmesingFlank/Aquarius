#pragma once

#include "Fluid.h"
#include "../Common/GpuCommons.h"


class Fluid_3D : public Fluid {
public:
	virtual glm::vec3 getCenter() = 0;
	virtual ~Fluid_3D() {};
};