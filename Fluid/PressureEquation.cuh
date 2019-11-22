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

__device__ __host__
struct PressureEquation3D {
	int termsIndex[7];
	float termsCoeff[7];
	unsigned char termCount = 0;
	float RHS;
	int x;
	int y;
	int z;
};