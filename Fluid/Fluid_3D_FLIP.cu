#include "Fluid_3D_FLIP.cuh"
#include "MAC_Grid_3D.cuh"
#include "../GpuCommons.h"
namespace Fluid_3D_FLIP {
	__global__  void transferToCellAccumPhase2(Cell3D* cells, Particle* particles, int particleCount, int sizeX, int sizeY, int sizeZ, float cellPhysicalSize) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= particleCount) return;


		Particle& particle = particles[index];
		float3 pos = particle.position;
		float3 vel = particle.velocity;
		int x = (pos.x / cellPhysicalSize);
		int y = (pos.y / cellPhysicalSize);
		int z = (pos.z / cellPhysicalSize);

		float3 velocityPos;
		float weight;

		Cell3D& thisCell = get3D(cells, x, y, z);
		Cell3D& rightCell = get3D(cells, x + 1, y, z);
		Cell3D& upCell = get3D(cells, x, y + 1, z);
		Cell3D& frontCell = get3D(cells, x, y, z + 1);

		thisCell.content = CONTENT_FLUID;

		velocityPos = make_float3(x, (y + 0.5), (z + 0.5)) * cellPhysicalSize;
		weight = trilinearHatKernel(pos - velocityPos, cellPhysicalSize);
		atomicAdd(&thisCell.weight.x, weight);
		atomicAdd(&thisCell.velocity.x, vel.x*weight);
		

		velocityPos = make_float3((x + 0.5), y, (z + 0.5)) * cellPhysicalSize;
		weight = trilinearHatKernel(pos - velocityPos, cellPhysicalSize);
		atomicAdd(&thisCell.weight.y, weight);
		atomicAdd(&thisCell.velocity.y, vel.y * weight);

		velocityPos = make_float3((x + 0.5), y + 0.5, z) * cellPhysicalSize;
		weight = trilinearHatKernel(pos - velocityPos, cellPhysicalSize);
		atomicAdd(&thisCell.weight.z, weight);
		atomicAdd(&thisCell.velocity.z, vel.z * weight);


		velocityPos = make_float3(x+1, (y + 0.5), (z + 0.5)) * cellPhysicalSize;
		weight = trilinearHatKernel(pos - velocityPos, cellPhysicalSize);
		atomicAdd(&rightCell.weight.x, weight);
		atomicAdd(&rightCell.velocity.x, vel.x * weight);


		velocityPos = make_float3((x + 0.5), y+1, (z + 0.5)) * cellPhysicalSize;
		weight = trilinearHatKernel(pos - velocityPos, cellPhysicalSize);
		atomicAdd(&upCell.weight.y, weight);
		atomicAdd(&upCell.velocity.y, vel.y * weight);

		velocityPos = make_float3((x + 0.5), y + 0.5, z+1) * cellPhysicalSize;
		weight = trilinearHatKernel(pos - velocityPos, cellPhysicalSize);
		atomicAdd(&frontCell.weight.z, weight);
		atomicAdd(&frontCell.velocity.z, vel.z * weight);
		

	}

	__global__  void transferToCellDividePhase2(Cell3D* cells, int sizeX, int sizeY, int sizeZ) {

		int index = blockIdx.x * blockDim.x + threadIdx.x;

		if (index >= sizeX * sizeY * sizeZ) return;

		int x = index / (sizeY * sizeZ);
		int y = (index - x * (sizeY * sizeZ)) / sizeZ;
		int z = index - x * (sizeY * sizeZ) - y * (sizeZ);

		Cell3D& thisCell = get3D(cells, x, y, z);


		if (thisCell.weight.x > 0) {
			thisCell.velocity.x /= thisCell.weight.x;
			thisCell.hasVelocityX = true;
		}

		if (thisCell.weight.y > 0) {
			thisCell.velocity.y /= thisCell.weight.y;
			thisCell.hasVelocityY = true;
		}

		if (thisCell.weight.z > 0) {
			thisCell.velocity.z /= thisCell.weight.z;
			thisCell.hasVelocityZ = true;
		}

		thisCell.newVelocity = thisCell.velocity;


	}

	__global__  void transferToCellAccumPhase(Cell3D* cells, int sizeX, int sizeY, int sizeZ, float cellPhysicalSize, int* cellStart, int* cellEnd,
		Particle* particles, int particleCount) {

		int index = blockIdx.x * blockDim.x + threadIdx.x;

		if (index >= sizeX * sizeY * sizeZ) return;

		int x = index / (sizeY * sizeZ);
		int y = (index - x * (sizeY * sizeZ)) / sizeZ;
		int z = index - x * (sizeY * sizeZ) - y * (sizeZ);

		Cell3D& thisCell = get3D(cells, x, y, z);


		int cellCount = (sizeX) * (sizeY) * (sizeZ);

		float3 xVelocityPos = make_float3(x, (y + 0.5), (z + 0.5)) * cellPhysicalSize;
		float3 yVelocityPos = make_float3((x + 0.5), y, (z + 0.5)) * cellPhysicalSize;
		float3 zVelocityPos = make_float3((x + 0.5), y + 0.5, z) * cellPhysicalSize;

		float3 thisVelocity = thisCell.velocity;
		float3 weight = make_float3(0, 0, 0);


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
								float3 pPosition = p.position;
								float3 pVelocity = p.velocity;
								float thisWeightX = trilinearHatKernel(pPosition - xVelocityPos, cellPhysicalSize);
								float thisWeightY = trilinearHatKernel(pPosition - yVelocityPos, cellPhysicalSize);
								float thisWeightZ = trilinearHatKernel(pPosition - zVelocityPos, cellPhysicalSize);


								thisVelocity.x += thisWeightX * pVelocity.x;
								thisVelocity.y += thisWeightY * pVelocity.y;
								thisVelocity.z += thisWeightZ * pVelocity.z;

								weight.x += thisWeightX;
								weight.y += thisWeightY;
								weight.z += thisWeightZ;
							}
						}
					}

				}
			}
		}

		thisCell.velocity = thisVelocity;
		thisCell.weight = weight;

	}


	__global__  void transferToCellDividePhase(Cell3D* cells, int sizeX, int sizeY, int sizeZ, float cellPhysicalSize, int* cellStart, int* cellEnd,
		Particle* particles, int particleCount) {

		int index = blockIdx.x * blockDim.x + threadIdx.x;

		if (index >= sizeX * sizeY * sizeZ) return;

		int x = index / (sizeY * sizeZ);
		int y = (index - x * (sizeY * sizeZ)) / sizeZ;
		int z = index - x * (sizeY * sizeZ) - y * (sizeZ);

		Cell3D& thisCell = get3D(cells, x, y, z);


		if (thisCell.weight.x > 0) {
			thisCell.velocity.x /= thisCell.weight.x;
			thisCell.hasVelocityX = true;
		}

		if (thisCell.weight.y > 0) {
			thisCell.velocity.y /= thisCell.weight.y;
			thisCell.hasVelocityY = true;
		}

		if (thisCell.weight.z > 0) {
			thisCell.velocity.z /= thisCell.weight.z;
			thisCell.hasVelocityZ = true;
		}

		thisCell.newVelocity = thisCell.velocity;

		for (int j = cellStart[index]; j <= cellEnd[index]; ++j) {
			if (j >= 0 && j < particleCount) {

				thisCell.content = CONTENT_FLUID;
				break;
			}
		}
	}



	__global__  void calcDensityImpl(Cell3D* cells, int sizeX, int sizeY, int sizeZ, float cellPhysicalSize, int* cellStart, int* cellEnd,
		Particle* particles, int particleCount) {

		int index = blockIdx.x * blockDim.x + threadIdx.x;

		if (index >= sizeX * sizeY * sizeZ) return;

		int x = index / (sizeY * sizeZ);
		int y = (index - x * (sizeY * sizeZ)) / sizeZ;
		int z = index - x * (sizeY * sizeZ) - y * (sizeZ);

		Cell3D& thisCell = get3D(cells, x, y, z);


		int cellCount = (sizeX) * (sizeY) * (sizeZ);

		float3 centerPos = make_float3((x + 0.5), (y + 0.5), (z + 0.5)) * cellPhysicalSize;

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



	__global__  void transferToParticlesImpl(Cell3D* cells, Particle* particles, int particleCount, int sizeX, int sizeY, int sizeZ, float cellPhysicalSize) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= particleCount) return;


		Particle& particle = particles[index];
		float3 newGridVelocity = MAC_Grid_3D::getPointNewVelocity(particle.position, cellPhysicalSize, sizeX, sizeY, sizeZ, cells);
		float3 oldGridVelocity = MAC_Grid_3D::getPointVelocity(particle.position, cellPhysicalSize, sizeX, sizeY, sizeZ, cells);
		float3 velocityChange = newGridVelocity - oldGridVelocity;
		particle.velocity += velocityChange; //FLIP

		//particle.velocity = newGridVelocity; //PIC


	}

	__global__  void moveParticlesImpl(float timeStep, Cell3D* cells, Particle* particles, int particleCount, int sizeX, int sizeY, int sizeZ, float cellPhysicalSize) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= particleCount) return;

		Particle& particle = particles[index];
		float3 beginPos = particle.position;


		float3 u1 = MAC_Grid_3D::getPointNewVelocity(beginPos, cellPhysicalSize, sizeX, sizeY, sizeZ, cells);
		float3 u2 = MAC_Grid_3D::getPointNewVelocity(beginPos + timeStep * u1 / 2, cellPhysicalSize, sizeX, sizeY, sizeZ, cells);
		float3 u3 = MAC_Grid_3D::getPointNewVelocity(beginPos + timeStep * u2 * 3 / 4, cellPhysicalSize, sizeX, sizeY, sizeZ, cells);

		float3 destPos = beginPos + timeStep * (u1 * 2 / 9 + u2 * 3 / 9 + u3 * 4 / 9);

		//destPos = beginPos+particle.velocity*timeStep;

		float epsilon = 1e-3;

		destPos.x = max(0.0 + epsilon, min(sizeX * cellPhysicalSize - epsilon, destPos.x));
		//destPos.y = max(0.0 + epsilon,  destPos.y );

		destPos.y = max(0.0 + epsilon, min(sizeY * cellPhysicalSize - epsilon, destPos.y));
		destPos.z = max(0.0 + epsilon, min(sizeZ * cellPhysicalSize - epsilon, destPos.z));


		particle.position = destPos;


	}

	__global__  void resetAllCells(Cell3D* cells, int sizeX, int sizeY, int sizeZ, float content) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;

		if (index >= sizeX * sizeY * sizeZ) return;

		int x = index / (sizeY * sizeZ);
		int y = (index - x * (sizeY * sizeZ)) / sizeZ;
		int z = index - x * (sizeY * sizeZ) - y * (sizeZ);


		Cell3D& thisCell = get3D(cells, x, y, z);


		thisCell.content = content;
		thisCell.velocity = make_float3(0, 0, 0);
		thisCell.newVelocity = make_float3(0, 0, 0);

		thisCell.hasVelocityX = false;
		thisCell.hasVelocityY = false;
		thisCell.hasVelocityZ = false;
	}







	Fluid::Fluid() {

	}



	void Fluid::simulationStep() {
		float thisTimeStep = 0.016f;

		transferToGrid();
		grid->updateFluidCount();

		applyGravity(thisTimeStep, *grid, gravitationalAcceleration);

		fixBoundary(*grid);

		calcDensity();

		computeDivergence(*grid, particlesPerCell);

		//solvePressure(thisTimeStep,*grid);

		solvePressureJacobi(thisTimeStep, *grid, 50);


		updateVelocityWithPressure(thisTimeStep, *grid);

		extrapolateVelocity(thisTimeStep, *grid);

		transferToParticles();

		moveParticles(thisTimeStep);
	}


	void Fluid::calcDensity() {
		calcDensityImpl << < numBlocksCell, numThreadsCell >> > (grid->cells, sizeX, sizeY, sizeZ, cellPhysicalSize, cellStart, cellEnd, particles, particleCount);
		CHECK_CUDA_ERROR("calcDensity");
	}

	


	void Fluid::transferToGrid() {
		performSpatialHashing2(particleIndices,particleHashes, particles, particlesCopy, particleCount, cellPhysicalSize, sizeX, sizeY, sizeZ, numBlocksParticle, numThreadsParticle, cellStart, cellEnd, cellCount);
		//performSpatialHashing(particleHashes, particles, particleCount, cellPhysicalSize, sizeX, sizeY, sizeZ, numBlocksParticle, numThreadsParticle, cellStart, cellEnd, cellCount);

		resetAllCells << < numBlocksCell, numThreadsCell >> > (grid->cells, sizeX, sizeY, sizeZ, CONTENT_AIR);
		cudaDeviceSynchronize();
		CHECK_CUDA_ERROR("reset all cells");

		//transferToCellAccumPhase2 << < numBlocksCell, numThreadsCell >> > (grid->cells, particles, particleCount, sizeX, sizeY, sizeZ, cellPhysicalSize);
		//transferToCellDividePhase2 << < numBlocksCell, numThreadsCell >> > (grid->cells, sizeX, sizeY, sizeZ);

		transferToCellAccumPhase << < numBlocksCell, numThreadsCell >> > (grid->cells, sizeX, sizeY, sizeZ, cellPhysicalSize, cellStart, cellEnd, particles, particleCount);
		transferToCellDividePhase << < numBlocksCell, numThreadsCell >> > (grid->cells, sizeX, sizeY, sizeZ, cellPhysicalSize, cellStart, cellEnd, particles, particleCount);
		CHECK_CUDA_ERROR("transfer to cell");
	}

	void Fluid::transferToParticles() {
		transferToParticlesImpl << <numBlocksParticle, numThreadsParticle >> > (grid->cells, particles, particleCount, sizeX, sizeY, sizeZ, cellPhysicalSize);
		cudaDeviceSynchronize();
		CHECK_CUDA_ERROR("transfer to particles");
	}

	void Fluid::moveParticles(float timeStep) {

		moveParticlesImpl << < numBlocksParticle, numThreadsParticle >> >
			(timeStep, grid->cells, particles, particleCount, sizeX, sizeY, sizeZ, cellPhysicalSize);
		cudaDeviceSynchronize();
		CHECK_CUDA_ERROR("move particles");
		return;

	}

	void Fluid::draw(const DrawCommand& drawCommand){
		skybox.draw(drawCommand);
		updatePositionsVBO << <numBlocksParticle, numThreadsParticle >> > (particles, pointSprites->positionsDevice, particleCount);
		cudaDeviceSynchronize();
		container->draw(drawCommand);

		float renderRadius = cellPhysicalSize / pow(particlesPerCell, 1.0 / 3.0);
		if (drawCommand.renderMode == RenderMode::Mesh) {
			mesher->mesh(particles, particlesCopy, particleHashes, particleIndices, meshRenderer->coordsDevice, make_float3(sizeX, sizeY, sizeZ) * cellPhysicalSize);
			cudaDeviceSynchronize();
			meshRenderer->draw(drawCommand);
		}
		else {
			pointSprites->draw(drawCommand, renderRadius, skybox.texSkyBox);
		}
		

		printGLError();

	}
	void Fluid::init(std::shared_ptr<FluidConfig> config) {
		//set everything to air first

		std::shared_ptr<FluidConfig3D> config3D = std::static_pointer_cast<FluidConfig3D, FluidConfig>(config);
		sizeX = config3D->sizeX;
		sizeY = config3D->sizeY;
		sizeZ = config3D->sizeZ;
		cellCount = (sizeX + 1) * (sizeY + 1) * (sizeZ + 1);
		cellPhysicalSize = 10.f / (float)sizeY;


		grid = std::make_shared<MAC_Grid_3D>(sizeX, sizeY, sizeZ, cellPhysicalSize);
		container = std::make_shared<Container>(glm::vec3(sizeX, sizeY, sizeZ) * cellPhysicalSize);

		Cell3D* cellsTemp = grid->copyCellsToHost();


		grid->fluidCount = 0;
		std::vector <Particle> particlesHost;

		for (const InitializationVolume& vol : config3D->initialVolumes) {
			if (vol.shapeType == ShapeType::Square) {
				float3 minPos = make_float3(vol.params[0], vol.params[1], vol.params[2]);
				float3 maxPos = make_float3(vol.params[3], vol.params[4], vol.params[5]);
				createSquareFluid(particlesHost, cellsTemp, minPos, maxPos, grid->fluidCount);
			}
			else if (vol.shapeType == ShapeType::Sphere) {
				float3 center = make_float3(vol.params[0], vol.params[1], vol.params[2]);
				float radius =  vol.params[3];
				createSphereFluid(particlesHost, cellsTemp, center,radius, grid->fluidCount);
			}
		}


		particleCount = particlesHost.size();

		grid->copyCellsToDevice(cellsTemp);
		delete[]cellsTemp;

		HANDLE_ERROR(cudaMalloc(&particles, particleCount * sizeof(Particle)));

		HANDLE_ERROR(cudaMemcpy(particles, particlesHost.data(), particleCount * sizeof(Particle),
			cudaMemcpyHostToDevice));

		HANDLE_ERROR(cudaMalloc(&particleHashes, particleCount * sizeof(*particleHashes)));
		HANDLE_ERROR(cudaMalloc(&cellStart, cellCount * sizeof(*cellStart)));
		HANDLE_ERROR(cudaMalloc(&cellEnd, cellCount * sizeof(*cellEnd)));

		numThreadsParticle = min(1024, particleCount);
		numBlocksParticle = divUp(particleCount, numThreadsParticle);

		numThreadsCell = min(1024, sizeX * sizeY * sizeZ);
		numBlocksCell = divUp(sizeX * sizeY * sizeZ, numThreadsCell);

		std::cout << numThreadsCell << std::endl << numBlocksCell << std::endl;

		fixBoundary(*grid);

		pointSprites = std::make_shared<PointSprites>(particleCount);

		HANDLE_ERROR(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, numBlocksCell * numThreadsCell * 1024));

		HANDLE_ERROR(cudaMalloc(&particleIndices, particleCount * sizeof(*particleIndices)));
		HANDLE_ERROR(cudaMalloc(&particlesCopy, particleCount * sizeof(Particle)));


		mesher = std::make_shared<Mesher>(sizeX, sizeY, sizeZ,particleCount,numBlocksParticle,numThreadsParticle);
		meshRenderer = std::make_shared<FluidMeshRenderer>(mesher->triangleCount);
	}


	void Fluid::createParticles(std::vector <Particle>& particlesHost, float3 centerPos, float tag ) {

		for (float dx = 0; dx <= 1; ++dx) {
			for (float dy = 0; dy <= 1; ++dy) {
				for (float dz = 0; dz <= 1; ++dz) {
					float3 subcellCenter = make_float3(dx - 0.5, dy - 0.5, dz - 0.5) * 0.5 * cellPhysicalSize + centerPos;
					float xBias = (random0to1() - 0.5f) * cellPhysicalSize*0.5;
					float yBias = (random0to1() - 0.5f) * cellPhysicalSize*0.5;
					float zBias = (random0to1() - 0.5f) * cellPhysicalSize*0.5;
					float3 particlePos = subcellCenter + make_float3(xBias, yBias, zBias);
					particlesHost.emplace_back(particlePos, tag);
				}
			}
		}
	}

	bool checkCoordValid(float3 c) {
		if (c.x < 0 || c.y < 0 || c.z < 0) {
			return false;
		}
		if (c.x > 1 || c.y> 1 || c.z > 1) {
			return false;
		}
		return true;
	}

	void Fluid::createSquareFluid(std::vector <Particle>& particlesHost, Cell3D* cellsTemp, float3 minPos, float3 maxPos, int startIndex) {
		int index = startIndex;

		if (!checkCoordValid(minPos) || !checkCoordValid(maxPos)) {
			return;
		}


		for (int z =  minPos.z* sizeZ; z < maxPos.z * sizeZ; ++z) {
			for (int x = minPos.x * sizeX; x < maxPos.x * sizeX; ++x) {
				for (int y = minPos.y * sizeY; y < maxPos.y * sizeY; ++y) {
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

		grid->fluidCount = index;
	}

	void Fluid::createSphereFluid(std::vector <Particle>& particlesHost, Cell3D* cellsTemp, float3 center, float radius,int startIndex) {
		int index = startIndex;
		if (!checkCoordValid(center)) {
			return;
		}
		for (int y = 0 * sizeY; y < 1 * sizeY; ++y) {
			for (int x = 0 * sizeX; x < 1 * sizeX; ++x) {
				for (int z = 0 * sizeZ; z < 1 * sizeZ; ++z) {
					if (pow(x - center.x * sizeX, 2) 
						+ pow(y - center.y * sizeY, 2) 
						+ pow(z - center.z * sizeZ, 2) 
						<= pow(radius * sizeY, 2)) {
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

		grid->fluidCount = index;
	}

}