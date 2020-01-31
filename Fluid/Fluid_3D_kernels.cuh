#pragma once

#include "../Common/GpuCommons.h"
#include "PressureEquation.cuh"
#include "MAC_Grid_3D.cuh"


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


__device__ __host__  float getNeibourCoefficient(int x, int y, int z,  float u, float& centerCoefficient, float& RHS, Cell3D* cells,
	int sizeX, int sizeY, int sizeZ);



__global__  void constructPressureEquations(Cell3D* cells, int sizeX, int sizeY, int sizeZ, PressureEquation3D* equations,  bool* hasNonZeroRHS);

__global__  void setPressure(Cell3D* cells, int sizeX, int sizeY, int sizeZ, double* pressureResult);


__global__  void updateVelocityWithPressureImpl(Cell3D* cells, int sizeX, int sizeY, int sizeZ);

__global__  void extrapolateVelocityByOne(Cell3D* cells, int sizeX, int sizeY, int sizeZ);


__global__  void computeDivergenceImpl(Cell3D* cells, int sizeX, int sizeY, int sizeZ, float cellPhysicalSize, float restParticlesPerCell);

__global__  void resetPressureImpl(Cell3D* cells, int sizeX, int sizeY, int sizeZ);

__global__  void jacobiImpl(Cell3D* cells, int sizeX, int sizeY, int sizeZ,  float cellPhysicalSize);

__global__  void precomputeNeighbors(Cell3D* cells, int sizeX, int sizeY, int sizeZ);


template<typename Particle>
__global__ inline void updatePositionsVBO(Particle* particles, float* positionsVBO, int particleCount) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= particleCount) return;

	float* base = positionsVBO + index * 7;
	Particle& particle = particles[index];


	base[0] = particle.position.x;
	base[1] = particle.position.y;
	base[2] = particle.position.z;
}

template<typename Particle>
__global__ inline void updatePositionsAndColorsVBO(Particle* particles, float* VBO, int particleCount,int phaseCount,float4* phaseToColor) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= particleCount) return;

	float* base = VBO + index * 7;
	Particle& particle = particles[index];

	base[0] = particle.position.x;
	base[1] = particle.position.y;
	base[2] = particle.position.z;

	float4 color = make_float4(0,0, 0, 0);
	for (int i = 0; i < phaseCount; ++i) {
		float fraction = particle.volumeFractions[i];
		float4 thisColor = phaseToColor[i];
		color += fraction * thisColor;
	}
	base[3] = color.x;
	base[4] = color.y;
	base[5] = color.z;
	base[6] = color.w;
}

__global__  void writeIndicesImpl(int* particleIndices, int particleCount);

template<typename Particle>
__global__  void applySortImpl(Particle* src, Particle* dest, int particleCount, int* particleIndices) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= particleCount) return;

	dest[index] = src[particleIndices[index]];
}



__global__ void findCellStartEndImpl(int* particleHashes,
	int* cellStart, int* cellEnd,
	int particleCount);
