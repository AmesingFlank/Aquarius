#pragma once

#include "../GpuCommons.h"
#include "PressureEquation.cuh"
#include "MAC_Grid_3D.cuh"
#include "Fluid_2D_kernels.cuh"

template<typename Particle>
__global__  inline void calcHashImpl(int* particleHashes,  // output
	Particle* particles,               // input: positions
	int particleCount,
	float cellPhysicalSize, int sizeX, int sizeY,int sizeZ) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= particleCount) return;

	Particle& p = particles[index];

	float3 pos = p.position;

	int x = pos.x / cellPhysicalSize;
	int y = pos.y / cellPhysicalSize;
	int z = pos.z / cellPhysicalSize;
	int hash = x * (sizeY*sizeZ)+y*(sizeZ)+z;


	particleHashes[index] = hash;
}


__global__  void applyGravityImpl(Cell3D* cells, int sizeX, int sizeY, int sizeZ, float timeStep, float gravitationalAcceleration);




__global__  void fixBoundaryX(Cell3D* cells, int sizeX, int sizeY, int sizeZ);
__global__  void fixBoundaryY(Cell3D* cells, int sizeX, int sizeY, int sizeZ);
__global__  void fixBoundaryZ(Cell3D* cells, int sizeX, int sizeY, int sizeZ);


__device__ __host__  float getNeibourCoefficient(int x, int y, int z, float dt_div_rho_div_dx, float u, float& centerCoefficient, float& RHS, Cell3D* cells,
	int sizeX, int sizeY, int sizeZ);



__global__  void constructPressureEquations(Cell3D* cells, int sizeX, int sizeY, int sizeZ, PressureEquation3D* equations, float dt_div_rho_div_dx, bool* hasNonZeroRHS);

__global__  void setPressure(Cell3D* cells, int sizeX, int sizeY, int sizeZ, double* pressureResult);


__global__  void updateVelocityWithPressureImpl(Cell3D* cells, int sizeX, int sizeY, int sizeZ, float dt_div_rho_div_dx);

__global__  void extrapolateVelocityByOne(Cell3D* cells, int sizeX, int sizeY, int sizeZ);


__global__  void computeDivergenceImpl(Cell3D* cells, int sizeX, int sizeY, int sizeZ, float cellPhysicalSize, float restParticlesPerCell);

__global__  void resetPressureImpl(Cell3D* cells, int sizeX, int sizeY, int sizeZ);

__global__  void jacobiImpl(Cell3D* cells, int sizeX, int sizeY, int sizeZ, float dt_div_rho_div_dx, float cellPhysicalSize);


template<typename Particle>
__global__ inline void updatePositionsVBO(Particle* particles, float* positionsVBO, int particleCount) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= particleCount) return;

	float* base = positionsVBO + index * 3;
	Particle& particle = particles[index];


	base[0] = particle.position.x;
	base[1] = particle.position.y;
	base[2] = particle.position.z;
}