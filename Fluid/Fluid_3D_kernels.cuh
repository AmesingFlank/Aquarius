#pragma once

#include "../Common/GpuCommons.h"
#include "PressureEquation.cuh"
#include "MAC_Grid_3D.cuh"
#include "VolumeData.cuh"


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


__global__  void applyGravityImpl(VolumeCollection volumes,int sizeX, int sizeY, int sizeZ, float timeStep, float gravitationalAcceleration);




__global__  void fixBoundaryX( VolumeCollection volumes, int sizeX, int sizeY, int sizeZ);
__global__  void fixBoundaryY( VolumeCollection volumes, int sizeX, int sizeY, int sizeZ);
__global__  void fixBoundaryZ( VolumeCollection volumes, int sizeX, int sizeY, int sizeZ);


__device__  float getNeibourCoefficient(int x, int y, int z,  float u, float& centerCoefficient, float& RHS,  VolumeCollection volumes, int sizeX, int sizeY, int sizeZ);



__global__  void constructPressureEquations( VolumeCollection volumes, int sizeX, int sizeY, int sizeZ, PressureEquation3D* equations,  bool* hasNonZeroRHS);

__global__  void setPressure( VolumeCollection volumes, int sizeX, int sizeY, int sizeZ, double* pressureResult);


__global__  void updateVelocityWithPressureImpl( VolumeCollection volumes, int sizeX, int sizeY, int sizeZ);

__global__  void extrapolateVelocityByOne( VolumeCollection volumes, int sizeX, int sizeY, int sizeZ);


__global__  void computeDivergenceImpl( VolumeCollection volumes, int sizeX, int sizeY, int sizeZ, float cellPhysicalSize, float restParticlesPerCell);


__global__  void jacobiImpl( VolumeCollection volumes, int sizeX, int sizeY, int sizeZ,  float cellPhysicalSize);


template<typename Particle>
__global__ inline void updatePositionsVBO(Particle* particles, float* positionsVBO, int particleCount,int stride) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= particleCount) return;

	float* base = positionsVBO + index * stride;
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
