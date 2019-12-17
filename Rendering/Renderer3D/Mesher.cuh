#pragma once

#include "../../Fluid/MAC_Grid_3D.cuh"
#include <memory>
#include "../../Fluid/SVD.cuh"

__global__
void marchingCubes(float* output, float* sdf, int sizeX_SDF, int sizeY_SDF, int sizeZ_SDF, float cellPhysicalSize_SDF, int sizeX_mesh, int sizeY_mesh, int sizeZ_mesh, float cellPhysicalSize_mesh,unsigned int* occupiedCellIndex);

template<typename Particle>
__global__
inline void computeSDF(Particle* particles, int particleCount,float particleRadius,int sizeX, int sizeY, int sizeZ, float cellPhysicalSize, int* cellStart,int* cellEnd,float* sdf,int* hasSDF,float3* meanXCell,float3* anistropy) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int cellCount = (sizeX) * (sizeY) * (sizeZ);

	if (index >= cellCount) return;

	int x = index / (sizeY * sizeZ);
	int y = (index - x * (sizeY * sizeZ)) / sizeZ;
	int z = index - x * (sizeY * sizeZ) - y * (sizeZ);

	
	// set to some postive value first. this means that, by default, all cells are not occupied.
	sdf[index] = cellPhysicalSize;

	float3 sumX = { 0,0,0 };
	float sumWeight = 0;

	float3 centerPos = make_float3(x,y,z) * cellPhysicalSize;

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
							float thisWeight = zhu05Kernel( (p.position - centerPos), cellPhysicalSize);
							sumWeight += thisWeight;

							sumX += thisWeight * p.position;

						}
					}
				}

			}
		}
	}

	if (sumWeight>0) {
		float3 meanX = sumX / sumWeight;
		float thisSDF = length(centerPos - meanX) - particleRadius;
		sdf[index] = thisSDF;
		hasSDF[index] = 1;
		meanXCell[index] = meanX;
	}

}



__global__
void extrapolateSDF(int sizeX, int sizeY, int sizeZ, float cellPhysicalSize, float particleRadius,float* sdf, int* hasSDF,float3* meanXCell);

__global__
void smoothSDF(int sizeX, int sizeY, int sizeZ, float cellPhysicalSize, float particleRadius, float* sdf, int* hasSDF, float sigma);




template<typename Particle>
__global__
inline void computeMeanXParticle(Particle* particles, int particleCount, float particleRadius, int sizeX, int sizeY, int sizeZ, float cellPhysicalSize, int* cellStart, int* cellEnd, float3* meanX) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int cellCount = (sizeX) * (sizeY) * (sizeZ);

	if (index >= particleCount) return;

	Particle& particle = particles[index];

	float3 pos = particle.position;
	int3 thisCell;

	thisCell.x = pos.x / cellPhysicalSize;
	thisCell.y = pos.y / cellPhysicalSize;
	thisCell.z = pos.z / cellPhysicalSize;

	float sumWeight = 0;

	float3 X = make_float3(0,0,0);

#pragma unroll
	for (int dx = -1; dx <= 1; ++dx) {
#pragma unroll
		for (int dy = -1; dy <= 1; ++dy) {
#pragma unroll
			for (int dz = -1; dz <= 1; ++dz) {
				int cell = (thisCell.x + dx) * sizeY * sizeZ + (thisCell.y + dy) * sizeZ + thisCell.z + dz;

				if (cell >= 0 && cell < cellCount) {

					for (int j = cellStart[cell]; j <= cellEnd[cell]; ++j) {
						if (j >= 0 && j < particleCount) {

							const Particle& p = particles[j];
							float thisWeight = pcaKernel(p.position - pos, cellPhysicalSize);
							sumWeight += thisWeight;


							X += thisWeight * p.position;
						}
					}
				}

			}
		}
	}

	X /= sumWeight;
	meanX[index] = X;


}


template<typename Particle>
__global__
inline void computeCovarianceMatrix(Particle* particles, int particleCount, float particleRadius, int sizeX, int sizeY, int sizeZ, float cellPhysicalSize, int* cellStart, int* cellEnd, Mat3x3* covMats,float3* meanX) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int cellCount = (sizeX) * (sizeY) * (sizeZ);

	if (index >= particleCount) return;

	Particle& particle = particles[index];

	float3 pos = meanX[index];
	int3 thisCell;

	thisCell.x = pos.x / cellPhysicalSize;
	thisCell.y = pos.y / cellPhysicalSize;
	thisCell.z = pos.z / cellPhysicalSize;

	float sumWeight = 0;
	
	Mat3x3 C;

#pragma unroll
	for (int dx = -1; dx <= 1; ++dx) {
#pragma unroll
		for (int dy = -1; dy <= 1; ++dy) {
#pragma unroll
			for (int dz = -1; dz <= 1; ++dz) {
				int cell = (thisCell.x + dx) * sizeY * sizeZ + (thisCell.y + dy) * sizeZ + thisCell.z + dz;

				if (cell >= 0 && cell < cellCount) {

					for (int j = cellStart[cell]; j <= cellEnd[cell]; ++j) {
						if (j >= 0 && j < particleCount) {

							const Particle& p = particles[j];
							float thisWeight = pcaKernel(p.position - pos, cellPhysicalSize);
							sumWeight += thisWeight;

							float3 xDiff = p.position - pos;

							Mat3x3 Cij;
							Cij.r0.x = xDiff.x * xDiff.x;
							Cij.r0.y = xDiff.x * xDiff.y;
							Cij.r0.z = xDiff.x * xDiff.z;

							Cij.r1.x = xDiff.y * xDiff.x;
							Cij.r1.y = xDiff.y * xDiff.y;
							Cij.r1.z = xDiff.y * xDiff.z;

							Cij.r2.x = xDiff.z * xDiff.x;
							Cij.r2.y = xDiff.z * xDiff.y;
							Cij.r2.z = xDiff.z * xDiff.z;

							C = C + Cij * thisWeight;

						}
					}
				}

			}
		}
	}

	C = C * (1.0 / sumWeight);

	covMats[index] = C;

}


template<typename Particle>
__global__
inline void computeAnistropy(Particle* particles, int particleCount, float particleRadius, int sizeX, int sizeY, int sizeZ, float cellPhysicalSize, int* cellStart, int* cellEnd, Mat3x3* covMats, float3* anistropy) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int cellCount = (sizeX) * (sizeY) * (sizeZ);

	if (index >= particleCount) return;

	Particle& particle = particles[index];

	float3 pos = particle.position;
	int3 thisCell;

	thisCell.x = pos.x / cellPhysicalSize;
	thisCell.y = pos.y / cellPhysicalSize;
	thisCell.z = pos.z / cellPhysicalSize;

	float sumWeight = 0;

	Mat3x3 C = covMats[index];
	
	float3 eVals;
	float3 v0;
	float3 v1;
	float3 v2;

	computeSVD(C, eVals, v0, v1, v2);
	float gammaMax = 4;
	float f = 0.75;
	
	float gamma = eVals.x / eVals.y;

	gamma = max(1.f, min(gamma, gammaMax));

	float result = f * (1.f - pow(1.f - pow((gammaMax - gamma) / (gammaMax - 1.f), 2), 3)) + 1.f - f;

	if (isnan(gamma) || isinf(gamma)) {
		printf("evs: %f %f\n", eVals.x, eVals.y);
		printf("gmma: %f\n", gamma);
		printf("result: %f\n", result);
	}

	anistropy[index].x = result;

}
template<typename Particle>
__global__
inline void computeSDF2(Particle* particles, int particleCount, float particleRadius, int sizeX, int sizeY, int sizeZ, float cellPhysicalSize, int* cellStart, int* cellEnd, float* sdf, int* hasSDF, float3* meanXCell, float3* anistropy) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int cellCount = (sizeX) * (sizeY) * (sizeZ);

	if (index >= cellCount) return;

	int x = index / (sizeY * sizeZ);
	int y = (index - x * (sizeY * sizeZ)) / sizeZ;
	int z = index - x * (sizeY * sizeZ) - y * (sizeZ);


	// set to some postive value first. this means that, by default, all cells are not occupied.
	sdf[index] = cellPhysicalSize;

	float3 sumX = { 0,0,0 };
	float sumWeight = 0;

	float3 centerPos = make_float3(x, y, z) * cellPhysicalSize;


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
							float thisWeight = Bcubic((p.position - centerPos) / anistropy[j].x, cellPhysicalSize);
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
		float thisSDF = length(meanX-centerPos) - particleRadius;
		sdf[index] = thisSDF;
		hasSDF[index] = 1;
		meanXCell[index] = meanX;
	}

}
struct Mesher {

	// two grids:
	// A coarse one for samping the SDF (resolution ~= MAC grid)
	// A fine one to do marching cubes (by sampling and interpolating the SDF on the coarse grid)

	int sizeX_SDF;
	int sizeY_SDF;
	int sizeZ_SDF;

	int sizeX_mesh = 256;
	int sizeY_mesh = 256;
	int sizeZ_mesh = 256;

	int cellCount_mesh = sizeX_mesh * sizeY_mesh * sizeZ_mesh;

	int cellCount_SDF;

	int numBlocksCell_SDF, numThreadsCell_SDF;

	int numBlocksCell_mesh, numThreadsCell_mesh;

	int triangleCount;

	int* cellStart;

	int* cellEnd;

	float* sdf;

	int* hasSDF;

	float3* meanXCell;

	unsigned int* occupiedCellIndex;

	int particleCount;
	int numBlocksParticle;
	int numThreadsParticle;

	Mat3x3* covMats;
	float3* meanXParticle;
	float3* anistropy;

	Mesher(int sizeX_sim, int sizeY_sim, int sizeZ_sim,int particleCount_,int numBlocksParticle_,int numThreadsParticle_){

		particleCount = particleCount_;
		numBlocksParticle = numBlocksParticle_;
		numThreadsParticle = numThreadsParticle_;

		float f = 1;

		sizeX_SDF = f*sizeX_sim + 2;
		sizeY_SDF = f*sizeY_sim + 2;
		sizeZ_SDF = f*sizeZ_sim + 2;

		cellCount_SDF = sizeX_SDF * sizeY_SDF * sizeZ_SDF;

		numThreadsCell_SDF = min(1024, cellCount_SDF);
		numBlocksCell_SDF = divUp(cellCount_SDF, numThreadsCell_SDF);

		numThreadsCell_mesh = min(1024, cellCount_mesh);
		numBlocksCell_mesh = divUp(cellCount_mesh, numThreadsCell_mesh);

		triangleCount = 5 * 1 << 22;

		std::cout << cellCount_mesh * 5 << std::endl;

		HANDLE_ERROR(cudaMalloc(&cellStart, cellCount_SDF * sizeof(*cellStart)));
		HANDLE_ERROR(cudaMalloc(&cellEnd, cellCount_SDF * sizeof(*cellEnd)));

		HANDLE_ERROR(cudaMalloc(&sdf, cellCount_SDF * sizeof(*sdf)));
		HANDLE_ERROR(cudaMalloc(&hasSDF, cellCount_SDF * sizeof(*hasSDF)));
		HANDLE_ERROR(cudaMalloc(&meanXCell, cellCount_SDF * sizeof(*meanXCell)));

		HANDLE_ERROR(cudaMalloc(&occupiedCellIndex, sizeof(unsigned int)));

		HANDLE_ERROR(cudaMalloc(&covMats, particleCount * sizeof(*covMats)));
		HANDLE_ERROR(cudaMalloc(&meanXParticle, particleCount * sizeof(*meanXParticle)));
		HANDLE_ERROR(cudaMalloc(&anistropy, particleCount * sizeof(*anistropy)));

	}

	template<typename Particle>
	void mesh(Particle* particles, Particle* particlesCopy,int* particleHashes, int* particleIndices, float* output, float3 containerSize) {
		
		HANDLE_ERROR(cudaMemset(output, 0, triangleCount * 3 * 3 * sizeof(float)));

		HANDLE_ERROR(cudaMemset(hasSDF, 0, cellCount_SDF * sizeof(*hasSDF)));

		HANDLE_ERROR(cudaMemset(occupiedCellIndex, 0, sizeof(unsigned int)));

		float cellPhysicalSize_SDF = containerSize.x / (float)(sizeX_SDF-1 );
		float particleRadius = cellPhysicalSize_SDF / 2;

		performSpatialHashing2(particleIndices, particleHashes, particles, particlesCopy, particleCount, cellPhysicalSize_SDF, sizeX_SDF, sizeY_SDF, sizeZ_SDF, numBlocksParticle, numThreadsParticle, cellStart, cellEnd, cellCount_SDF);




		computeMeanXParticle << <numBlocksParticle, numThreadsParticle >> > (particles, particleCount, particleRadius, sizeX_SDF, sizeY_SDF, sizeZ_SDF, cellPhysicalSize_SDF, cellStart, cellEnd, meanXParticle);
		CHECK_CUDA_ERROR("compute meanX");

		computeCovarianceMatrix << <numBlocksParticle, numThreadsParticle >> > (particles, particleCount, particleRadius, sizeX_SDF, sizeY_SDF, sizeZ_SDF, cellPhysicalSize_SDF, cellStart, cellEnd, covMats, meanXParticle);
		CHECK_CUDA_ERROR("compute cov mat");


		computeAnistropy << <numBlocksParticle, numThreadsParticle >> > (particles, particleCount, particleRadius, sizeX_SDF, sizeY_SDF, sizeZ_SDF, cellPhysicalSize_SDF, cellStart, cellEnd, covMats,anistropy);
		CHECK_CUDA_ERROR("compute anistropy");




		computeSDF2 << <numBlocksCell_SDF, numThreadsCell_SDF >> > (particles, particleCount, particleRadius, sizeX_SDF, sizeY_SDF, sizeZ_SDF, cellPhysicalSize_SDF, cellStart, cellEnd, sdf, hasSDF,meanXCell,anistropy);
		CHECK_CUDA_ERROR("compute sdf");
		
		
		
		extrapolateSDF << <numBlocksCell_SDF, numThreadsCell_SDF >> > (sizeX_SDF, sizeY_SDF, sizeZ_SDF, cellPhysicalSize_SDF, particleRadius, sdf, hasSDF, meanXCell);
		CHECK_CUDA_ERROR("extrapolate sdf");

		for (int i = 0; i < 0; ++i) {
			smoothSDF << <numBlocksCell_SDF, numThreadsCell_SDF >> > (sizeX_SDF, sizeY_SDF, sizeZ_SDF, cellPhysicalSize_SDF, particleRadius, sdf, hasSDF, 1);
			CHECK_CUDA_ERROR("smooth sdf");
		}

		float cellPhysicalSize_mesh = containerSize.x / (float)(sizeX_mesh-1);

		marchingCubes<<<numBlocksCell_mesh,numThreadsCell_mesh >>>(output, sdf, sizeX_SDF, sizeY_SDF, sizeZ_SDF, cellPhysicalSize_SDF,sizeX_mesh, sizeY_mesh,sizeZ_mesh, cellPhysicalSize_mesh,occupiedCellIndex);
		CHECK_CUDA_ERROR("marching cubes");


		

	}
	
};