#pragma once

#include "../../Common/GpuCommons.h"
#include "../PressureEquation.cuh"
#include "MAC_Grid_2D.cuh"

template<typename Particle>
__global__ inline void calcHashImpl(int* particleHashes,  // output
	Particle* particles,               // input: positions
	int particleCount,
	float cellPhysicalSize, int sizeX, int sizeY) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= particleCount) return;

	Particle& p = particles[index];

	float2 pos = p.position;

	int x = pos.x / cellPhysicalSize;
	int y = pos.y / cellPhysicalSize;
	int hash = x * (sizeY) + y;

	particleHashes[index] = hash;
}


__global__ void applyGravityImpl(Cell2D* cells, int sizeX, int sizeY, float timeStep, float gravitationalAcceleration);

__global__ void fixBoundaryX(Cell2D* cells, int sizeX, int sizeY);

__global__ void fixBoundaryY(Cell2D* cells, int sizeX, int sizeY);

__device__ __host__ float getNeibourCoefficient(int x, int y, float dt_div_rho_div_dx, float u, float& centerCoefficient, float& RHS, Cell2D* cells,int sizeX, int sizeY);

__global__  void constructPressureEquations(Cell2D* cells, int sizeX, int sizeY, PressureEquation2D* equations, float dt_div_rho_div_dx, bool* hasNonZeroRHS);

__global__ void setPressure(Cell2D* cells, int sizeX, int sizeY, double* pressureResult);

__global__ void updateVelocityWithPressureImpl(Cell2D* cells, int sizeX, int sizeY, float dt_div_rho_div_dx);

__global__ void extrapolateVelocityByOne(Cell2D* cells, int sizeX, int sizeY);

__global__ void drawCellImpl(Cell2D* cells, int sizeX, int sizeY, unsigned char* image);

template<typename Particle>
__global__ inline void drawParticleImpl(float containerSizeX, float containerSizeY, Particle* particles, int particleCount,
	unsigned char* image, int imageSizeX, int imageSizeY) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= particleCount) return;

	Particle& particle = particles[index];

	int x = (float)imageSizeX * particle.position.x / containerSizeX;
	int y = (float)imageSizeY * particle.position.y / containerSizeY;
	unsigned char* base = image + (y * imageSizeX + x) * 4;

	if (particle.kind == 0) {
		base[0] = 0;
		base[1] = 0;
		base[2] = 255;
	}
	else {
		base[0] = 255;
		base[1] = 0;
		base[2] = 0;
	}
	base[3] = 255;

}


__global__ void computeDivergenceImpl(Cell2D* cells, int sizeX, int sizeY, float cellPhysicalSize);

__global__ void resetPressureImpl(Cell2D* cells, int sizeX, int sizeY);

__global__ void jacobiImpl(Cell2D* cells, int sizeX, int sizeY, float dt_div_rho_div_dx, float cellPhysicalSize);