#pragma once

#include "../Common/GpuCommons.h"
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


__global__  void applyGravityImpl(VolumeCollection volumes,int sizeX, int sizeY, int sizeZ, float timeStep, float3 gravitationalAcceleration);




__global__  void fixBoundaryX( VolumeCollection volumes, int sizeX, int sizeY, int sizeZ);
__global__  void fixBoundaryY( VolumeCollection volumes, int sizeX, int sizeY, int sizeZ);
__global__  void fixBoundaryZ( VolumeCollection volumes, int sizeX, int sizeY, int sizeZ);


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
__global__ inline void updatePositionsAndPhasesVBO(Particle* particles, float* VBO, int particleCount,int stride) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= particleCount) return;

	float* base = VBO + index * stride;
	Particle& particle = particles[index];

	base[0] = particle.position.x;
	base[1] = particle.position.y;
	base[2] = particle.position.z;

	


	base[3] = particle.volumeFractions.x;
	base[4] = particle.volumeFractions.y;
	base[5] = particle.volumeFractions.z;
	base[6] = particle.volumeFractions.w;

	


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





__global__  void diffusionJacobiImpl(VolumeCollection volumes, int sizeX, int sizeY, int sizeZ, float lambda);