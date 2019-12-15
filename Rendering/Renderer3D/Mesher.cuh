#pragma once

#include "../../Fluid/MAC_Grid_3D.cuh"
#include <memory>


__global__
void marchingCubes(float* output, float* sdf, int sizeX, int sizeY, int sizeZ, float cellPhysicalSize);

template<typename Particle>
__global__
inline void computeSDF(Particle* particles, int particleCount,float particleRadius,int sizeX, int sizeY, int sizeZ, float cellPhysicalSize, int* cellStart,int* cellEnd,float* sdf,int* hasSDF,float3* meanXs) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int cellCount = (sizeX) * (sizeY) * (sizeZ);

	if (index >= cellCount) return;

	int x = index / (sizeY * sizeZ);
	int y = (index - x * (sizeY * sizeZ)) / sizeZ;
	int z = index - x * (sizeY * sizeZ) - y * (sizeZ);

	sdf[index] = cellPhysicalSize; // set to some postive value first. this means that, by default, all cells are not occupied.

	if (x == 0 || y == 0 || z == 0 || x == sizeX - 1 || y == sizeY - 1 || z == sizeZ - 1) {
		return;
	}

	float3 sumX = { 0,0,0 };
	float sumWeight = 0;

	float3 centerPos = make_float3((x - 0.5), (y - 0.5), (z - 0.5)) * cellPhysicalSize;


#pragma unroll
	for (int dx = -1; dx <= 1; ++dx) {
#pragma unroll
		for (int dy = -1; dy <= 1; ++dy) {
#pragma unroll
			for (int dz = -1; dz <= 1; ++dz) {
				int cell = (x + dx) * sizeY * sizeZ + (y + dy) * sizeZ + z + dz;

				if (cell >= 0 && cell < cellCount) {

					for (int j = cellStart[cell]; j <= cellEnd[cell]; ++j) {
						if (j >= 0 && j < particleCount) {

							const Particle& p = particles[j];
							float thisWeight = trilinearHatKernel(p.position - centerPos, cellPhysicalSize);
							sumWeight += thisWeight;

							sumX += thisWeight * p.position;

						}
					}
				}

			}
		}
	}

	if (sumWeight != 0) {
		float3 meanX = sumX / sumWeight;
		float thisSDF = length(centerPos - meanX) - particleRadius;
		sdf[index] = thisSDF;
		hasSDF[index] = 1;
		meanXs[index] = meanX;
	}

}



__global__
void extrapolateSDF(int sizeX, int sizeY, int sizeZ, float cellPhysicalSize, float particleRadius,float* sdf, int* hasSDF,float3* meanXs);


struct Mesher {

	int sizeX_SDF;
	int sizeY_SDF;
	int sizeZ_SDF;

	int cellCount_SDF;

	int numBlocksCell_SDF, numThreadsCell_SDF;

	int triangleCount;

	int* cellStart;

	int* cellEnd;

	float* sdf;

	int* hasSDF;

	float3* meanXs;

	Mesher(int sizeX_sim, int sizeY_sim, int sizeZ_sim){

		sizeX_SDF = sizeX_sim + 2;
		sizeY_SDF = sizeY_sim + 2;
		sizeZ_SDF = sizeZ_sim + 2;

		cellCount_SDF = sizeX_SDF * sizeY_SDF * sizeZ_SDF;
		numThreadsCell_SDF = min(1024, cellCount_SDF);
		numBlocksCell_SDF = divUp(cellCount_SDF, numThreadsCell_SDF);

		triangleCount = cellCount_SDF * 5;

		HANDLE_ERROR(cudaMalloc(&cellStart, cellCount_SDF * sizeof(*cellStart)));
		HANDLE_ERROR(cudaMalloc(&cellEnd, cellCount_SDF * sizeof(*cellEnd)));

		HANDLE_ERROR(cudaMalloc(&sdf, cellCount_SDF * sizeof(*sdf)));
		HANDLE_ERROR(cudaMalloc(&hasSDF, cellCount_SDF * sizeof(*hasSDF)));
		HANDLE_ERROR(cudaMalloc(&meanXs, cellCount_SDF * sizeof(*meanXs)));
	}

	template<typename Particle>
	void mesh(Particle* particles, Particle* particlesCopy,int* particleHashes, int* particleIndices, int particleCount, int numBlocksParticle,int numThreadsParticle, float* output, float3 containerSize) {
		
		HANDLE_ERROR(cudaMemset(output, 0, triangleCount * 3 * 3 * sizeof(float)));

		HANDLE_ERROR(cudaMemset(hasSDF, 0, cellCount_SDF * sizeof(*hasSDF)));

		float cellPhysicalSize = containerSize.x / (float)(sizeX_SDF -2);
		performSpatialHashing2(particleIndices, particleHashes, particles, particlesCopy, particleCount, cellPhysicalSize, sizeX_SDF, sizeY_SDF, sizeZ_SDF, numBlocksParticle, numThreadsParticle, cellStart, cellEnd, cellCount_SDF);
		computeSDF << <numBlocksCell_SDF, numThreadsCell_SDF >> > (particles, particleCount, cellPhysicalSize, sizeX_SDF, sizeY_SDF, sizeZ_SDF, cellPhysicalSize, cellStart, cellEnd, sdf, hasSDF,meanXs);
		extrapolateSDF << <numBlocksCell_SDF, numThreadsCell_SDF >> > (sizeX_SDF, sizeY_SDF, sizeZ_SDF, cellPhysicalSize, cellPhysicalSize, sdf, hasSDF, meanXs);

		marchingCubes<<<numBlocksCell_SDF,numThreadsCell_SDF >>>(output, sdf, sizeX_SDF, sizeY_SDF, sizeZ_SDF, cellPhysicalSize);
	}
	
};