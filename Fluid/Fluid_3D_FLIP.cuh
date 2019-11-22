#pragma once


#include "MAC_Grid_3D.cuh"
#include "SPD_Solver.h"
#include <vector>
#include <utility>
#include "../GpuCommons.h"
#include "Fluid_3D.cuh"
#include <unordered_map>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include "FLuid_3D_common.cuh"
#include "FLuid_3D_kernels.cuh"
#include "../Rendering/Renderer3D/PointSprites.h"
#include "../Rendering/Renderer3D/Container.h"
#include "../Rendering/Renderer3D/Skybox.h"

namespace Fluid_3D_FLIP{
	template<typename Particle>
	__global__ inline void transferToCell(Cell3D* cells, int sizeX, int sizeY, int sizeZ,float cellPhysicalSize, int* cellStart, int* cellEnd,
		Particle* particles, int particleCount) {

		int index = blockIdx.x * blockDim.x + threadIdx.x;

		if (index >= sizeX * sizeY * sizeZ) return;

		int x = index / (sizeY * sizeZ);
		int y = (index - x * (sizeY * sizeZ)) / sizeZ;
		int z = index - x * (sizeY * sizeZ) - y * (sizeZ);

		Cell3D& thisCell = get3D(cells, x, y,z);


		int cellCount = (sizeX) * (sizeY) * (sizeZ);

		float3 xVelocityPos = make_float3(x, (y + 0.5), (z + 0.5)) * cellPhysicalSize;
		float3 yVelocityPos = make_float3((x + 0.5), y, (z + 0.5)) * cellPhysicalSize;
		float3 zVelocityPos = make_float3((x + 0.5), y + 0.5, z) * cellPhysicalSize;


		float totalWeightX = 0;
		float totalWeightY = 0;
		float totalWeightZ = 0;

		if (isinf(thisCell.velocity.y)) {
			printf("inf y in start of transfertocell\n");
		}
		if (isnan(thisCell.velocity.y)) {
			printf("nah y in start of transfertocell\n");
		}

		
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

								

								//float yBefore = thisCell.velocity.y;

								const Particle& p = particles[j];
								float thisWeightX = trilinearHatKernel(p.position - xVelocityPos, cellPhysicalSize);
								float thisWeightY = trilinearHatKernel(p.position - yVelocityPos, cellPhysicalSize);
								float thisWeightZ = trilinearHatKernel(p.position - zVelocityPos, cellPhysicalSize);



								if (isinf(thisWeightY)) {
									printf("inf y in weight\n");
								}
								if (isnan(thisWeightY)) {
									printf("nah y in weight\n");
								}

								if (isinf(p.velocity.y)) {
									printf("inf y in p in transfercell\n");
								}
								if (isnan(p.velocity.y)) {
									printf("nah y in p in transfercell\n");
								}

								thisCell.velocity.x += thisWeightX * p.velocity.x;
								thisCell.velocity.y += thisWeightY * p.velocity.y;
								thisCell.velocity.z += thisWeightZ * p.velocity.z;

								if (isinf(thisCell.velocity.y)) {
									printf("inf y after adding p in transfertocell %f %f %d %d %d\n",p.velocity.y,thisWeightY,x,y,z);
								}
								if (isnan(thisCell.velocity.y)) {
									printf("nah y after adding p in transfertocell\n");
								}

								totalWeightX += thisWeightX;
								totalWeightY += thisWeightY;
								totalWeightZ += thisWeightZ;
							}
						}
					}

				}
			}
		}
		
		if (totalWeightX > 0) {
			thisCell.velocity.x /= totalWeightX;
			thisCell.hasVelocityX = true;
		}
			
		if (totalWeightY > 0) {
			thisCell.velocity.y /= totalWeightY;
			thisCell.hasVelocityY = true;
		}
			
		if (totalWeightZ > 0) {
			thisCell.velocity.z /= totalWeightZ;
			thisCell.hasVelocityZ = true;
		}
			
		thisCell.newVelocity = thisCell.velocity;

		for (int j = cellStart[index]; j <= cellEnd[index]; ++j) {
			if (j >= 0 && j < particleCount) {

				thisCell.content = CONTENT_FLUID;

				const Particle& p = particles[j];

				if (p.kind > 0) {
					thisCell.fluid1Count++;
				}
				else {
					thisCell.fluid0Count++;
				}
			}
		}
	}


	template<typename Particle>
	__global__ inline void calcDensityImpl(Cell3D* cells, int sizeX, int sizeY, int sizeZ, float cellPhysicalSize, int* cellStart, int* cellEnd,
		Particle* particles, int particleCount) {

		int index = blockIdx.x * blockDim.x + threadIdx.x;

		if (index >= sizeX * sizeY * sizeZ) return;

		int x = index / (sizeY * sizeZ);
		int y = (index - x * (sizeY * sizeZ)) / sizeZ;
		int z = index - x * (sizeY * sizeZ) - y * (sizeZ);

		Cell3D& thisCell = get3D(cells, x, y, z);


		int cellCount = (sizeX) * (sizeY) * (sizeZ);

		float3 centerPos = make_float3((x+0.5), (y + 0.5), (z + 0.5)) * cellPhysicalSize;

		float totalWeight = 0;

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
								totalWeight += thisWeight;

							}
						}
					}

				}
			}
		}

		thisCell.density = totalWeight;
	}


	template<typename Particle>
	__global__ inline void transferToParticlesImpl(Cell3D* cells, Particle* particles, int particleCount, int sizeX, int sizeY, int sizeZ,float cellPhysicalSize) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= particleCount) return;


		Particle& particle = particles[index];
		float3 newGridVelocity = MAC_Grid_3D::getPointNewVelocity(particle.position, cellPhysicalSize, sizeX, sizeY,sizeZ, cells);
		float3 oldGridVelocity = MAC_Grid_3D::getPointVelocity(particle.position, cellPhysicalSize, sizeX, sizeY,sizeZ, cells);
		float3 velocityChange = newGridVelocity - oldGridVelocity;
		particle.velocity += velocityChange; //FLIP

		//particle.velocity = newGridVelocity; //PIC


		if (isinf(particle.velocity.y)) {
			printf("inf y in transfertoParticle\n");
		}
		if (isnan(particle.velocity.y)) {
			printf("nah y in transfertoParticle\n");
		}

	}

	template<typename Particle>
	__global__ inline void moveParticlesImpl(float timeStep, Cell3D* cells, Particle* particles, int particleCount, int sizeX, int sizeY, int sizeZ,float cellPhysicalSize) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= particleCount) return;

		Particle& particle = particles[index];
		float3 beginPos = particle.position;


		float3 u1 = MAC_Grid_3D::getPointNewVelocity(beginPos, cellPhysicalSize, sizeX, sizeY, sizeZ,cells);
		float3 u2 = MAC_Grid_3D::getPointNewVelocity(beginPos + timeStep * u1 / 2, cellPhysicalSize, sizeX, sizeY,sizeZ, cells);
		float3 u3 = MAC_Grid_3D::getPointNewVelocity(beginPos + timeStep * u2 * 3 / 4, cellPhysicalSize, sizeX, sizeY,sizeZ, cells);

		float3 destPos = beginPos + timeStep * (u1 * 2 / 9 + u2 * 3 / 9 + u3 * 4 / 9);

		//destPos = beginPos+particle.velocity*timeStep;

		
		destPos.x = max(0.0 + 1e-6, min(sizeX * cellPhysicalSize - 1e-6, destPos.x));
		//destPos.y = max(0.0 + 1e-6,  destPos.y );

		destPos.y = max(0.0 + 1e-6, min(sizeY * cellPhysicalSize - 1e-6, destPos.y));
		destPos.z = max(0.0 + 1e-6, min(sizeZ * cellPhysicalSize - 1e-6, destPos.z));
		

		particle.position = destPos;

		if (isinf(particle.velocity.y)) {
			printf("inf y in moveParticle\n");
		}
		if (isnan(particle.velocity.y)) {
			printf("nah y in moveParticle\n");
		}

	}

	__global__ inline void resetAllCells(Cell3D* cells, int sizeX, int sizeY, int sizeZ,float content) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;

		if (index >= sizeX * sizeY * sizeZ) return;

		int x = index / (sizeY * sizeZ);
		int y = (index - x * (sizeY * sizeZ)) / sizeZ;
		int z = index - x * (sizeY * sizeZ) - y * (sizeZ);


		Cell3D& thisCell = get3D(cells, x, y,z);


		thisCell.content = content;
		thisCell.velocity = make_float3(0,0, 0);
		thisCell.newVelocity = make_float3(0,0, 0);
		thisCell.fluid0Count = 0;
		thisCell.fluid1Count = 0;
		thisCell.hasVelocityX = false;
		thisCell.hasVelocityY = false;
		thisCell.hasVelocityZ = false;
	}

	__device__ __host__ struct Particle {
		float3 position = make_float3(0, 0,0);
		float kind = 0;
		float3 velocity = make_float3(0, 0,0);

		__device__ __host__
			Particle() {

		}
		Particle(float3 pos) :position(pos) {

		}
		Particle(float3 pos, float tag) :position(pos), kind(tag) {

		}
	};

	class Fluid :Fluid_3D {
	public:
		const int sizeX = 32;
		const int sizeY = 32;
		const int sizeZ = 32;



		const int cellCount = (sizeX + 1) * (sizeY + 1)*(sizeZ + 1);


		const float cellPhysicalSize = 10.f / (float)sizeY;
		const float gravitationalAcceleration = 9.8;
		const float density = 1;
		MAC_Grid_3D grid = MAC_Grid_3D(sizeX, sizeY,sizeZ, cellPhysicalSize);

		Particle* particles;
		int particleCount;

		int numThreadsParticle, numBlocksParticle;
		int numThreadsCell, numBlocksCell;

		int particlesPerCell = 16;

		int* particleHashes;
		int* cellStart;
		int* cellEnd;

		Container container = Container(glm::vec3(sizeX, sizeY, sizeZ)*cellPhysicalSize);

		Skybox skybox = Skybox("resources/Park2/", ".jpg");

		PointSprites* pointSprites;

		Fluid() {
			init();


		}

		void init() {
			

			//set everything to air first

			Cell3D* cellsTemp = grid.copyCellsToHost();


			grid.fluidCount = 0;
			std::vector <Particle> particlesHost;
			createSquareFluid(particlesHost, cellsTemp);
			createSphereFluid(particlesHost, cellsTemp, grid.fluidCount);
			particleCount = particlesHost.size();

			grid.copyCellsToDevice(cellsTemp);
			delete[]cellsTemp;

			HANDLE_ERROR(cudaMalloc(&particles, particleCount * sizeof(Particle)));
			
			HANDLE_ERROR(cudaMemcpy(particles, particlesHost.data(), particleCount * sizeof(Particle),
				cudaMemcpyHostToDevice));

			HANDLE_ERROR(cudaMalloc(&particleHashes, particleCount * sizeof(*particleHashes)));
			HANDLE_ERROR(cudaMalloc(&cellStart, cellCount * sizeof(*cellStart)));
			HANDLE_ERROR(cudaMalloc(&cellEnd, cellCount * sizeof(*cellEnd)));

			numThreadsParticle = min(1024, particleCount);
			numBlocksParticle = divUp(particleCount, numThreadsParticle);

			numThreadsCell = min(1024, sizeX * sizeY * sizeZ );
			numBlocksCell = divUp(sizeX * sizeY * sizeZ, numThreadsCell);

			std::cout << numThreadsCell << std::endl << numBlocksCell << std::endl;

			fixBoundary(grid);

			pointSprites = new PointSprites(particleCount);

			HANDLE_ERROR(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, numBlocksCell * numThreadsCell * 1024));

		}

		virtual void simulationStep() override {
			float thisTimeStep = 0.016f;

			transferToGrid();
			grid.updateFluidCount();

			applyGravity(thisTimeStep, grid, gravitationalAcceleration);

			fixBoundary(grid);

			calcDensity();

			//solvePressure(thisTimeStep,grid);

			solvePressureJacobi(thisTimeStep, grid, particlesPerCell,20);


			updateVelocityWithPressure(thisTimeStep, grid);

			extrapolateVelocity(thisTimeStep, grid);

			transferToParticles();

			moveParticles(thisTimeStep);
		}


		void calcDensity() {
			calcDensityImpl << < numBlocksCell, numThreadsCell >> > (grid.cells, sizeX, sizeY, sizeZ, cellPhysicalSize, cellStart, cellEnd, particles, particleCount);
			CHECK_CUDA_ERROR("calcDensity");
		}

		

		void transferToGrid() {
			performSpatialHashing(particleHashes, particles, particleCount, cellPhysicalSize, sizeX, sizeY,sizeZ, numBlocksParticle, numThreadsParticle, cellStart, cellEnd, cellCount);

			resetAllCells << < numBlocksCell, numThreadsCell >> > (grid.cells, sizeX, sizeY,sizeZ, CONTENT_AIR);
			cudaDeviceSynchronize();
			CHECK_CUDA_ERROR("reset all cells");

			transferToCell << < numBlocksCell, numThreadsCell >> > (grid.cells, sizeX, sizeY, sizeZ,cellPhysicalSize, cellStart, cellEnd, particles, particleCount);
			CHECK_CUDA_ERROR("transfer to cell");
		}

		void transferToParticles() {
			transferToParticlesImpl << <numBlocksParticle, numThreadsParticle >> > (grid.cells, particles, particleCount, sizeX, sizeY, sizeZ,cellPhysicalSize);
			cudaDeviceSynchronize();
			CHECK_CUDA_ERROR("transfer to particles");
		}

		void moveParticles(float timeStep) {

			moveParticlesImpl << < numBlocksParticle, numThreadsParticle >> >
				(timeStep, grid.cells, particles, particleCount, sizeX, sizeY, sizeZ,cellPhysicalSize);
			cudaDeviceSynchronize();
			CHECK_CUDA_ERROR("move particles");
			return;

		}

		virtual void draw(const DrawCommand& drawCommand) override {
			skybox.draw(drawCommand);
			updatePositionsVBO<<<numBlocksParticle,numThreadsParticle>>>(particles, pointSprites->positionsDevice, particleCount);
			cudaDeviceSynchronize();
			pointSprites->draw(drawCommand,cellPhysicalSize/2);
			container.draw(drawCommand);
			printGLError();

		}
		void createParticles(std::vector <Particle>& particlesHost, float3 centerPos, float tag = 0) {
			for (int particle = 0; particle < particlesPerCell; ++particle) {
				float xBias = (random0to1() - 0.5f) * cellPhysicalSize;
				float yBias = (random0to1() - 0.5f) * cellPhysicalSize;
				float zBias = (random0to1() - 0.5f) * cellPhysicalSize;
				//xBias = 0;yBias=0;zBias=0;
				float3 particlePos = centerPos + make_float3(xBias, yBias, zBias);

				particlesHost.emplace_back(particlePos, tag);
			}
		}

		void createSquareFluid(std::vector <Particle>& particlesHost, Cell3D* cellsTemp, int startIndex = 0) {
			int index = startIndex;
			for (int z = 0 * sizeZ; z < 1 * sizeZ; ++z) {
				for (int x = 0 * sizeX; x < 1 * sizeX; ++x) {
					for (int y = 0.0 * sizeY; y < 0.3 * sizeY; ++y) {
						Cell3D& thisCell = get3D(cellsTemp,x,y,z);

						thisCell.velocity = make_float3(0, 0, 0);
						thisCell.newVelocity = make_float3(0, 0,0);
						thisCell.content = CONTENT_FLUID;
						thisCell.fluidIndex = index;
						++index;
						float3 thisPos = MAC_Grid_3D::getPhysicalPos(x, y,z, cellPhysicalSize);
						createParticles(particlesHost, thisPos, 0);
					}
				}
			}

			grid.fluidCount = index;
		}

		void createSphereFluid(std::vector <Particle>& particlesHost, Cell3D* cellsTemp, int startIndex = 0) {
			int index = startIndex;
			for (int y = 0 * sizeY; y < 1 * sizeY; ++y) {
				for (int x = 0 * sizeX; x < 1 * sizeX; ++x) {
					for (int z = 0 * sizeZ; z < 1 * sizeZ; ++z) {
						if (pow(x - 0.5 * sizeX, 2) + pow(y - 0.7 * sizeY, 2) + pow(z-0.5*sizeZ,2) <= pow(0.2 * sizeY, 2)) {
						//if ( x==20 && y==20 && z==20 ) {
							Cell3D& thisCell = get3D(cellsTemp, x, y, z);

							thisCell.velocity = make_float3(0, 0, 0);
							thisCell.newVelocity = make_float3(0, 0, 0);
							thisCell.content = CONTENT_FLUID;
							thisCell.fluidIndex = index;
							++index;
							float3 thisPos = MAC_Grid_3D::getPhysicalPos(x, y, z, cellPhysicalSize);
							createParticles(particlesHost, thisPos, 0);
						}
					}
				}
			}

			grid.fluidCount = index;
		}

	};
}
