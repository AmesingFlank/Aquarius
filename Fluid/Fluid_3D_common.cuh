#pragma once
#include "Fluid_3D_kernels.cuh"

template<typename Particle>
void inline performSpatialHashing(int* particleHashes, Particle* particles, int particleCount, float cellPhysicalSize, float sizeX, float sizeY, float sizeZ,int numBlocksParticle, int numThreadsParticle, int* cellStart, int* cellEnd, int cellCount) {
	calcHashImpl<Particle> << < numBlocksParticle, numThreadsParticle >> > (particleHashes, particles, particleCount, cellPhysicalSize, sizeX, sizeY,sizeZ);
	CHECK_CUDA_ERROR("calc hash");

	thrust::sort_by_key(thrust::device, particleHashes, particleHashes + particleCount, particles);

	HANDLE_ERROR(cudaMemset(cellStart, 255, cellCount * sizeof(*cellStart)));
	HANDLE_ERROR(cudaMemset(cellEnd, 255, cellCount * sizeof(*cellEnd)));
	findCellStartEndImpl << < numBlocksParticle, numThreadsParticle >> > (particleHashes, cellStart, cellEnd, particleCount);
	CHECK_CUDA_ERROR("find cell start end");

}


void  applyGravity(float timeStep, MAC_Grid_3D& grid, float gravitationalAcceleration);

void  fixBoundary(MAC_Grid_3D& grid);

void  computeDivergence(MAC_Grid_3D& grid, float restParticlesPerCell);
void  solvePressureJacobi(float timeStep, MAC_Grid_3D& grid, int iterations);

void  solvePressure(float timeStep, MAC_Grid_3D& grid);

void  updateVelocityWithPressure(float timeStep, MAC_Grid_3D& grid);

void  extrapolateVelocity(float timeStep, MAC_Grid_3D& grid);