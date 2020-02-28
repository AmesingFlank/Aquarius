#pragma once

#include "../../Fluid/MAC_Grid_3D.cuh"
#include <memory>
#include <thread>

__global__
void marchingCubes(float* output, int sizeX_SDF, int sizeY_SDF, int sizeZ_SDF, float cellPhysicalSize_SDF, int sizeX_mesh, int sizeY_mesh, int sizeZ_mesh, float cellPhysicalSize_mesh,unsigned int* occupiedCellIndex,cudaTextureObject_t sdfTexture);


__global__
void smoothSDF(int sizeX, int sizeY, int sizeZ, float cellPhysicalSize, float particleRadius, cudaSurfaceObject_t sdfSurface, int* hasSDF, float sigma);



// not using anistropy
template<typename Particle>
__global__
inline void computeSDF(Particle* particles, int particleCount,float particleRadius,int sizeX, int sizeY, int sizeZ, float cellPhysicalSize, int* cellStart,int* cellEnd,int* hasSDF,float3* meanXCell,cudaSurfaceObject_t sdfSurface) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int cellCount = (sizeX) * (sizeY) * (sizeZ);

	if (index >= cellCount) return;

	int x = index / (sizeY * sizeZ);
	int y = (index - x * (sizeY * sizeZ)) / sizeZ;
	int z = index - x * (sizeY * sizeZ) - y * (sizeZ);


	// set to some postive value first. this means that, by default, all cells are not occupied.
	surf3Dwrite<float>(cellPhysicalSize, sdfSurface, x * sizeof(float), y, z);
	hasSDF[index] = 0;

	float3 sumX = { 0,0,0 };
	float sumWeight = 0;

	float3 centerPos = make_float3(x - 1, y - 1, z - 1) * cellPhysicalSize;

	float kernelRadius = cellPhysicalSize * 2.f  ;
	float kernelRadius3 = kernelRadius * kernelRadius * kernelRadius;

#pragma unroll
	for (int dx = -2; dx <= 1; ++dx) {
#pragma unroll
		for (int dy = -2; dy <= 1; ++dy) {
#pragma unroll
			for (int dz = -2; dz <= 1; ++dz) {
				int cell = (x + dx - 1) * (sizeY - 2) * (sizeZ - 2) + (y + dy - 1) * (sizeZ - 2) + z + dz - 1;

				if (cell >= 0 && cell < cellCount) {

					for (int j = cellStart[cell]; j <= cellEnd[cell]; ++j) {
						if (j >= 0 && j < particleCount) {

							const Particle& p = particles[j];
							//float thisWeight = cubic_spline_kernel(length(p.position - centerPos), kernelRadius,kernelRadius3);
							float thisWeight = dunfanKernel( p.position - centerPos, kernelRadius);
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
		float thisSDF = length(meanX - centerPos) - particleRadius;
		//thisSDF = zhu05Kernel(make_float3(particleRadius,0,0),cellPhysicalSize) - sumWeight ; //blobby
		surf3Dwrite<float>(thisSDF, sdfSurface, x * sizeof(float), y, z);
		hasSDF[index] = 1;
		meanXCell[index] = meanX;
	}

}



__global__
void extrapolateSDF(int sizeX, int sizeY, int sizeZ, float cellPhysicalSize, float particleRadius,int* hasSDF,float3* meanXCell,cudaSurfaceObject_t sdfSurface);



struct Mesher {

	// two grids:
	// A coarse one for samping the SDF (resolution ~= MAC grid)
	// A fine one to do marching cubes (by sampling and interpolating the SDF on the coarse grid)

	int sizeX_SDF;
	int sizeY_SDF;
	int sizeZ_SDF;

	int sizeX_mesh = 128;
	int sizeY_mesh = 128;
	int sizeZ_mesh = 128;

	int cellCount_mesh = sizeX_mesh * sizeY_mesh * sizeZ_mesh;

	int cellCount_SDF;

	int numBlocksCell_SDF, numThreadsCell_SDF;

	int numBlocksCell_mesh, numThreadsCell_mesh;

	int triangleCount;

	int* cellStart;

	int* cellEnd;

	cudaArray* sdfTextureArray;
	cudaTextureObject_t sdfTexture;
	cudaSurfaceObject_t sdfSurface;

	int* hasSDF;

	float3* meanXCell;

	unsigned int* occupiedCellIndex;

	int particleCount;
	int numBlocksParticle;
	int numThreadsParticle;



	float cellPhysicalSize_SDF;
	float particleRadius;
	float cellPhysicalSize_mesh;

	Mesher(float3 containerSize,float particleSpacing,int particleCount_,int numBlocksParticle_,int numThreadsParticle_){

		particleCount = particleCount_;
		numBlocksParticle = numBlocksParticle_;
		numThreadsParticle = numThreadsParticle_;


		cellPhysicalSize_SDF = particleSpacing ;
		particleRadius = particleSpacing;

		std::cout << "mesher particle radius " << particleRadius << std::endl;
		cellPhysicalSize_mesh = containerSize.x / (float)(sizeX_mesh - 2);

		sizeX_SDF = 3 + containerSize.x / cellPhysicalSize_SDF;
		sizeY_SDF = 3 + containerSize.y / cellPhysicalSize_SDF;
		sizeZ_SDF = 3 + containerSize.z / cellPhysicalSize_SDF;

		std::cout << "mesher sizeX_SDF " << sizeX_SDF << std::endl;

		cellCount_SDF = sizeX_SDF * sizeY_SDF * sizeZ_SDF;

		numThreadsCell_SDF = min(1024, cellCount_SDF);
		numBlocksCell_SDF = divUp(cellCount_SDF, numThreadsCell_SDF);

		numThreadsCell_mesh = min(1024, cellCount_mesh);
		numBlocksCell_mesh = divUp(cellCount_mesh, numThreadsCell_mesh);

		triangleCount = 5 * 1 << 22;

		std::cout << cellCount_mesh * 5 << std::endl;

		HANDLE_ERROR(cudaMalloc(&cellStart, cellCount_SDF * sizeof(*cellStart)));
		HANDLE_ERROR(cudaMalloc(&cellEnd, cellCount_SDF * sizeof(*cellEnd)));

		HANDLE_ERROR(cudaMalloc(&hasSDF, cellCount_SDF * sizeof(*hasSDF)));
		HANDLE_ERROR(cudaMalloc(&meanXCell, cellCount_SDF * sizeof(*meanXCell)));

		HANDLE_ERROR(cudaMalloc(&occupiedCellIndex, sizeof(unsigned int)));

		


		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
		cudaExtent extent = { sizeX_SDF,sizeY_SDF,sizeZ_SDF };
		HANDLE_ERROR(cudaMalloc3DArray(&sdfTextureArray, &channelDesc, extent, cudaArraySurfaceLoadStore));
		
		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = sdfTextureArray;

		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.addressMode[0] = cudaAddressModeBorder;
		texDesc.addressMode[1] = cudaAddressModeBorder;
		texDesc.addressMode[2] = cudaAddressModeBorder;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = 1;

		HANDLE_ERROR(cudaCreateTextureObject(&sdfTexture, &resDesc, &texDesc, nullptr));

		HANDLE_ERROR(cudaCreateSurfaceObject(&sdfSurface, &resDesc));


	}


	template<typename Particle>
	void mesh(Particle*& particles, Particle*& particlesCopy, int* particleHashes, int* particleIndices, float* output) {

		HANDLE_ERROR(cudaMemset(occupiedCellIndex, 0, sizeof(unsigned int)));
		HANDLE_ERROR(cudaMemset(hasSDF, 0, cellCount_SDF * sizeof(*hasSDF)));


		HANDLE_ERROR(cudaMemset(output, 0, triangleCount * 3 * 6 * sizeof(float)));

		performSpatialHashing2(particleIndices, particleHashes, particles, particlesCopy, particleCount, cellPhysicalSize_SDF, sizeX_SDF - 2, sizeY_SDF - 2, sizeZ_SDF - 2, numBlocksParticle, numThreadsParticle, cellStart, cellEnd, cellCount_SDF);

		

		computeSDF << <numBlocksCell_SDF, numThreadsCell_SDF >> > (particles, particleCount, particleRadius, sizeX_SDF, sizeY_SDF, sizeZ_SDF, cellPhysicalSize_SDF, cellStart, cellEnd, hasSDF, meanXCell, sdfSurface);
		CHECK_CUDA_ERROR("compute sdf");





		extrapolateSDF << <numBlocksCell_SDF, numThreadsCell_SDF >> > (sizeX_SDF, sizeY_SDF, sizeZ_SDF, cellPhysicalSize_SDF, particleRadius, hasSDF, meanXCell, sdfSurface);
		CHECK_CUDA_ERROR("extrapolate sdf");

		for (int i = 0; i < 0; ++i) {
			smoothSDF << <numBlocksCell_SDF, numThreadsCell_SDF >> > (sizeX_SDF, sizeY_SDF, sizeY_SDF, cellPhysicalSize_SDF, particleRadius, sdfSurface, hasSDF, 10);
			CHECK_CUDA_ERROR("smooth sdf");
		}



		marchingCubes << <numBlocksCell_mesh, numThreadsCell_mesh >> > (output, sizeX_SDF, sizeY_SDF, sizeZ_SDF, cellPhysicalSize_SDF, sizeX_mesh, sizeY_mesh, sizeZ_mesh, cellPhysicalSize_mesh, occupiedCellIndex, sdfTexture);
		CHECK_CUDA_ERROR("marching cubes");

		cudaDeviceSynchronize();
	}

	~Mesher() {
		cudaDeviceSynchronize();
		HANDLE_ERROR(cudaFree(cellStart ));
		HANDLE_ERROR(cudaFree(cellEnd ));

		HANDLE_ERROR(cudaFree(hasSDF ));
		HANDLE_ERROR(cudaFree(meanXCell ));

		HANDLE_ERROR(cudaFree(occupiedCellIndex));

		HANDLE_ERROR(cudaFreeArray(sdfTextureArray));
	}
};