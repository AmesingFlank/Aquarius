#pragma once
#include "Fluid_3D_kernels.cuh"

template<typename Particle>
void inline performSpatialHashing(int* particleHashes, Particle* particles, int particleCount, float cellPhysicalSize, float sizeX, float sizeY, float sizeZ,int numBlocksParticle, int numThreadsParticle, int* cellStart, int* cellEnd, int cellCount) {
	calcHashImpl<Particle> << < numBlocksParticle, numThreadsParticle >> > (particleHashes, particles, particleCount, cellPhysicalSize, sizeX, sizeY,sizeZ);
	//cudaDeviceSynchronize();
	CHECK_CUDA_ERROR("calc hash");

	thrust::sort_by_key(thrust::device, particleHashes, particleHashes + particleCount, particles);

	HANDLE_ERROR(cudaMemset(cellStart, 255, cellCount * sizeof(*cellStart)));
	HANDLE_ERROR(cudaMemset(cellEnd, 255, cellCount * sizeof(*cellEnd)));
	findCellStartEndImpl << < numBlocksParticle, numThreadsParticle >> > (particleHashes, cellStart, cellEnd, particleCount);
	CHECK_CUDA_ERROR("find cell start end");

}

// this is faster than performSpatialHashing
// the function sorts the indices of the particles, instead of the particles themselves
// the result is then written into result, which is assumed to be already allocated.
// the two Particle* are passed as references, and is swapped, so that after the function finishes, 
// particles become result.
template<typename Particle>
void inline performSpatialHashing2(int* particleIndices, int* particleHashes, Particle*& particles, Particle*& result,int particleCount, float cellPhysicalSize, float sizeX, float sizeY, float sizeZ, int numBlocksParticle, int numThreadsParticle, int* cellStart, int* cellEnd, int cellCount) {
	
	writeIndicesImpl << <numBlocksParticle, numThreadsParticle >> > (particleIndices, particleCount);
	calcHashImpl << < numBlocksParticle, numThreadsParticle >> > (particleHashes, particles, particleCount, cellPhysicalSize, sizeX, sizeY, sizeZ);
	//cudaDeviceSynchronize();
	CHECK_CUDA_ERROR("calc hash");

	thrust::sort_by_key(thrust::device, particleHashes, particleHashes + particleCount, particleIndices);
	cudaDeviceSynchronize();


	applySortImpl << < numBlocksParticle, numThreadsParticle >> > (particles, result, particleCount, particleIndices);
	CHECK_CUDA_ERROR("apply sort");

	std::swap(particles, result);


	HANDLE_ERROR(cudaMemset(cellStart, 255, cellCount * sizeof(*cellStart)));
	HANDLE_ERROR(cudaMemset(cellEnd, 255, cellCount * sizeof(*cellEnd)));
	findCellStartEndImpl << < numBlocksParticle, numThreadsParticle >> > (particleHashes, cellStart, cellEnd, particleCount);
	CHECK_CUDA_ERROR("find cell start end");
	cudaDeviceSynchronize();


}


void  applyGravity(float timeStep, MAC_Grid_3D& grid, float gravitationalAcceleration);

void  fixBoundary(MAC_Grid_3D& grid);

void  computeDivergence(MAC_Grid_3D& grid, float restParticlesPerCell);
void  solvePressureJacobi(float timeStep, MAC_Grid_3D& grid, int iterations);

void  solvePressure(float timeStep, MAC_Grid_3D& grid);

void  updateVelocityWithPressure(float timeStep, MAC_Grid_3D& grid);

void  extrapolateVelocity(float timeStep, MAC_Grid_3D& grid);