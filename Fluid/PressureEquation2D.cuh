#pragma once

#include "../GpuCommons.h"
__device__ __host__
struct PressureEquation2D {
	int termsIndex[5];
	float termsCoeff[5];
	unsigned char termCount = 0;
	float RHS;
	int x;
	int y;
};