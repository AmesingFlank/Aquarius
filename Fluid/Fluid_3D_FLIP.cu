#include "Fluid_3D_FLIP.cuh"
#include "MAC_Grid_3D.cuh"
#include "../Common/GpuCommons.h"
#include <thread>
namespace Fluid_3D_FLIP {

	__global__  void transferToCellAccumPhase(VolumeCollection volumes, int sizeX, int sizeY, int sizeZ, float cellPhysicalSize, int* cellStart, int* cellEnd,Particle* particles, int particleCount) {



		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		int z = blockIdx.z * blockDim.z + threadIdx.z;

		if (x >= sizeX || y >= sizeY || z >= sizeZ) return;


		int cellCount = (sizeX) * (sizeY) * (sizeZ);

		float3 xVelocityPos = make_float3(x, (y + 0.5), (z + 0.5)) * cellPhysicalSize;
		float3 yVelocityPos = make_float3((x + 0.5), y, (z + 0.5)) * cellPhysicalSize;
		float3 zVelocityPos = make_float3((x + 0.5), y + 0.5, z) * cellPhysicalSize;

		float4 thisVelocity = make_float4(0, 0, 0,0);
		float4 weight = make_float4(0, 0, 0,0);


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

		volumes.velocityAccumWeight.writeSurface<float4>(weight, x, y, z);
		volumes.velocity.writeSurface<float4>(thisVelocity, x, y, z);

	}


	__global__  void transferToCellDividePhase(VolumeCollection volumes, int sizeX, int sizeY, int sizeZ, float cellPhysicalSize, int* cellStart, int* cellEnd, Particle* particles, int particleCount) {

		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		int z = blockIdx.z * blockDim.z + threadIdx.z;

		if (x >= sizeX || y >= sizeY || z >= sizeZ) return;

		int index = x * (sizeY* sizeZ) + y * (sizeZ)+z;



		float4 weight = volumes.velocityAccumWeight.readSurface<float4>(x, y, z);
		int4 hasVelocity = volumes.hasVelocity.readSurface<int4>(x, y, z);
		float4 velocity = volumes.velocity.readSurface<float4>(x, y, z);

		if (weight.x > 0) {
			velocity.x /= weight.x;
			hasVelocity.x = true;
		}

		if (weight.y > 0) {
			velocity.y /= weight.y;
			hasVelocity.y = true;
		}

		if (weight.z > 0) {
			velocity.z /= weight.z;
			hasVelocity.z = true;
		}

		volumes.velocity.writeSurface<float4>(velocity,x,y,z);
		volumes.newVelocity.writeSurface<float4>(velocity, x, y, z);
		volumes.hasVelocity.writeSurface<int4>(hasVelocity, x, y, z);

		

		for (int j = cellStart[index]; j <= cellEnd[index]; ++j) {
			if (j >= 0 && j < particleCount) {

				volumes.content.writeSurface<int>(CONTENT_FLUID, x, y, z);
				break;
			}
		}
	}



	__global__  void countParticlesInCellsImpl(VolumeCollection volumes, int sizeX, int sizeY, int sizeZ, float cellPhysicalSize, int* cellStart, int* cellEnd, Particle* particles, int particleCount) {

		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		int z = blockIdx.z * blockDim.z + threadIdx.z;

		if (x >= sizeX || y >= sizeY || z >= sizeZ) return;

		int index = x * (sizeY * sizeZ) + y * (sizeZ)+z;



		int count = cellEnd[index] - cellStart[index];



		volumes.particleCount.writeSurface<int>(count, x, y,z);
	}



	__global__  void transferToParticlesImpl(VolumeCollection volumes, Particle* particles, int particleCount, int sizeX, int sizeY, int sizeZ, float cellPhysicalSize,float FLIP_coeff) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= particleCount) return;


		Particle& particle = particles[index];
		float3 newGridVelocity = MAC_Grid_3D::getPointNewVelocity(particle.position, cellPhysicalSize, sizeX, sizeY, sizeZ,  volumes);
		float3 oldGridVelocity = MAC_Grid_3D::getPointVelocity(particle.position, cellPhysicalSize, sizeX, sizeY, sizeZ,volumes);
		float3 velocityChange = newGridVelocity - oldGridVelocity;
		
		float3 FLIP = particle.velocity + velocityChange;
		float3 PIC = newGridVelocity;

		

		particle.velocity = FLIP_coeff * FLIP + (1.0f - FLIP_coeff) * PIC;

		//particle.velocity = PIC;

	}

	__global__  void moveParticlesImpl(VolumeCollection volumes, Particle* particles, int particleCount, int sizeX, int sizeY, int sizeZ, float cellPhysicalSize,float timeStep) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= particleCount) return;

		Particle& particle = particles[index];
		float3 beginPos = particle.position;


		float3 u1 = MAC_Grid_3D::getPointNewVelocity(beginPos, cellPhysicalSize, sizeX, sizeY, sizeZ,volumes);
		float3 u2 = MAC_Grid_3D::getPointNewVelocity(beginPos + timeStep * u1 / 2, cellPhysicalSize, sizeX, sizeY, sizeZ,  volumes);
		float3 u3 = MAC_Grid_3D::getPointNewVelocity(beginPos + timeStep * u2 * 3 / 4, cellPhysicalSize, sizeX, sizeY, sizeZ, volumes);

		float3 destPos = beginPos + timeStep * (u1 * 2 / 9 + u2 * 3 / 9 + u3 * 4 / 9);

		//destPos = beginPos+particle.velocity*timeStep;

		float minDistanceFromWall = cellPhysicalSize / 4;

		float3 gridPhysicalSize = make_float3(sizeX, sizeY, sizeZ) * cellPhysicalSize;

		float bounce = -0.5;

		if (destPos.x < minDistanceFromWall) {
			destPos.x = minDistanceFromWall;
			particle.velocity.x *= bounce;;
		}

		if (destPos.x > gridPhysicalSize.x - minDistanceFromWall) {
			destPos.x = gridPhysicalSize.x - minDistanceFromWall;
			particle.velocity.x *= bounce;;
		}

		if (destPos.y < minDistanceFromWall) {
			destPos.y = minDistanceFromWall;
			particle.velocity.y *= bounce;;
		}

		if (destPos.y > gridPhysicalSize.y - minDistanceFromWall) {
			destPos.y = gridPhysicalSize.y - minDistanceFromWall;
			particle.velocity.y *= bounce;;
		}

		if (destPos.z < minDistanceFromWall) {
			destPos.z = minDistanceFromWall;
			particle.velocity.z *= bounce;;
		}

		if (destPos.z > gridPhysicalSize.z - minDistanceFromWall) {
			destPos.z = gridPhysicalSize.z - minDistanceFromWall;
			particle.velocity.z *= bounce;;
		}


		particle.position = destPos;


	}

	__global__  void resetCellImpl(VolumeCollection volumes, int sizeX, int sizeY, int sizeZ, float content) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;

		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		int z = blockIdx.z * blockDim.z + threadIdx.z;

		if (x >= sizeX || y >= sizeY || z >= sizeZ) return;


		volumes.content.writeSurface<int>(CONTENT_AIR, x, y, z);

		volumes.hasVelocity.writeSurface<int4>(make_int4(0, 0, 0, 0), x, y, z);
		volumes.velocity.writeSurface<float4>(make_float4(0, 0, 0, 0), x,y,z);
		volumes.newVelocity.writeSurface<float4>(make_float4(0, 0, 0, 0), x, y, z);
	}







	Fluid::Fluid() {

	}



	void Fluid::simulationStep() {

		transferToGrid();
		//grid->updateFluidCount();

		applyGravity(timestep, *grid, gravitationalAcceleration);

		fixBoundary(*grid);

		countParticlesInCells();

		computeDivergence(*grid, particlesPerCell);

		//solvePressure(timestep,*grid);

		solvePressureJacobi(timestep, *grid, 100);


		updateVelocityWithPressure(timestep, *grid);

		extrapolateVelocity(timestep, *grid);

		transferToParticles();

		moveParticles(timestep);
	}


	void Fluid::countParticlesInCells() {
		countParticlesInCellsImpl << < grid->cudaGridSize, grid->cudaBlockSize >> > ( grid->volumes, sizeX, sizeY, sizeZ, cellPhysicalSize, cellStart, cellEnd, particles, particleCount);
		CHECK_CUDA_ERROR("calcDensity");
	}

	


	void Fluid::transferToGrid() {
		performSpatialHashing2(particleIndices,particleHashes, particles, particlesCopy, particleCount, cellPhysicalSize, sizeX, sizeY, sizeZ, numBlocksParticle, numThreadsParticle, cellStart, cellEnd, cellCount);
		//performSpatialHashing(particleHashes, particles, particleCount, cellPhysicalSize, sizeX, sizeY, sizeZ, numBlocksParticle, numThreadsParticle, cellStart, cellEnd, cellCount);

		resetCellImpl <<< grid->cudaGridSize, grid->cudaBlockSize >>> ( grid->volumes, sizeX, sizeY, sizeZ, CONTENT_AIR);
		cudaDeviceSynchronize();
		CHECK_CUDA_ERROR("reset all cells");


		transferToCellAccumPhase <<< grid->cudaGridSize,grid->cudaBlockSize >> > ( grid->volumes, sizeX, sizeY, sizeZ, cellPhysicalSize, cellStart, cellEnd, particles, particleCount);
		transferToCellDividePhase << < grid->cudaGridSize, grid->cudaBlockSize >> > ( grid->volumes, sizeX, sizeY, sizeZ, cellPhysicalSize, cellStart, cellEnd, particles, particleCount);
		CHECK_CUDA_ERROR("transfer to cell");
	}

	void Fluid::transferToParticles() {
		transferToParticlesImpl << <numBlocksParticle, numThreadsParticle >> > ( grid->volumes, particles, particleCount, sizeX, sizeY, sizeZ, cellPhysicalSize,0.95);
		cudaDeviceSynchronize();
		CHECK_CUDA_ERROR("transfer to particles");
	}

	void Fluid::moveParticles(float timeStep) {

		moveParticlesImpl << < numBlocksParticle, numThreadsParticle >> >
			(grid->volumes, particles, particleCount, sizeX, sizeY, sizeZ, cellPhysicalSize, timeStep);
		cudaDeviceSynchronize();
		CHECK_CUDA_ERROR("move particles");
		return;

	}

	void Fluid::draw(const DrawCommand& drawCommand){
		skybox.draw(drawCommand);
		
		//container->draw(drawCommand);

		if (drawCommand.renderMode == RenderMode::Mesh) {


			mesher->mesh(particles, particlesCopy, particleHashes, particleIndices, meshRenderer->coordsDevice);

			cudaDeviceSynchronize();

			meshRenderer->draw(drawCommand,skybox.texSkyBox);

			drawInk(drawCommand);
		}
		else {
			float renderRadius = cellPhysicalSize / pow(particlesPerCell, 1.0 / 3.0);

			updatePositionsVBO << <numBlocksParticle, numThreadsParticle >> > (particles, pointSprites->positionsDevice, particleCount,pointSprites->stride);
			cudaDeviceSynchronize();
			pointSprites->draw(drawCommand, renderRadius, skybox.texSkyBox);
		}
		

		printGLError();

	}
	void Fluid::init(std::shared_ptr<FluidConfig> config) {
		//set everything to air first

		std::shared_ptr<FluidConfig3D> config3D = std::static_pointer_cast<FluidConfig3D, FluidConfig>(config);
		fluidConfig = config3D;
		sizeX = config3D->sizeX;
		sizeY = config3D->sizeY;
		sizeZ = config3D->sizeZ;
		cellCount = (sizeX + 1) * (sizeY + 1) * (sizeZ + 1);
		cellPhysicalSize = 10.f / (float)sizeY;


		grid = std::make_shared<MAC_Grid_3D>(sizeX, sizeY, sizeZ, cellPhysicalSize);

		container = std::make_shared<Container>(glm::vec3(sizeX, sizeY, sizeZ) * cellPhysicalSize);



		grid->fluidCount = 0;
		std::vector <Particle> particlesHost;

		for (const InitializationVolume& vol : config3D->initialVolumes) {
			if (vol.shapeType == ShapeType::Square) {
				float3 minPos = make_float3(vol.params[0], vol.params[1], vol.params[2]);
				float3 maxPos = make_float3(vol.params[3], vol.params[4], vol.params[5]);
				createSquareFluid(particlesHost, minPos, maxPos, grid->fluidCount);
			}
			else if (vol.shapeType == ShapeType::Sphere) {
				float3 center = make_float3(vol.params[0], vol.params[1], vol.params[2]);
				float radius = vol.params[3];
				createSphereFluid(particlesHost, center, radius, grid->fluidCount);
			}
		}


		particleCount = particlesHost.size();

		

		HANDLE_ERROR(cudaMalloc(&particles, particleCount * sizeof(Particle)));

		HANDLE_ERROR(cudaMemcpy(particles, particlesHost.data(), particleCount * sizeof(Particle),
			cudaMemcpyHostToDevice));

		HANDLE_ERROR(cudaMalloc(&particleHashes, particleCount * sizeof(*particleHashes)));
		HANDLE_ERROR(cudaMalloc(&cellStart, cellCount * sizeof(*cellStart)));
		HANDLE_ERROR(cudaMalloc(&cellEnd, cellCount * sizeof(*cellEnd)));

		numThreadsParticle = min(1024, particleCount);
		numBlocksParticle = divUp(particleCount, numThreadsParticle);


		fixBoundary(*grid);

		pointSprites = std::make_shared<PointSprites>(particleCount);

		HANDLE_ERROR(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, grid->cellCount * 1024));

		HANDLE_ERROR(cudaMalloc(&particleIndices, particleCount * sizeof(*particleIndices)));
		HANDLE_ERROR(cudaMalloc(&particlesCopy, particleCount * sizeof(Particle)));


		mesher = std::make_shared<Mesher>(make_float3(sizeX, sizeY, sizeZ) * cellPhysicalSize, cellPhysicalSize / 2, particleCount, numBlocksParticle, numThreadsParticle);
		meshRenderer = std::make_shared<FluidMeshRenderer>(mesher->triangleCount);

		initInkRenderer();

		transferToGrid();

	}


	int Fluid::createParticlesAt(std::vector <Particle>& particlesHost, float3 centerPos, std::function<bool(float3)> filter,float particleSpacing) {
		int createdCount = 0;
		for (float dx = 0; dx <= 1; ++dx) {
			for (float dy = 0; dy <= 1; ++dy) {
				for (float dz = 0; dz <= 1; ++dz) {
					float3 subcellCenter = make_float3(dx - 0.5, dy - 0.5, dz - 0.5) * particleSpacing+ centerPos;

					float xJitter = (random0to1() - 0.5f) ;
					float yJitter = (random0to1() - 0.5f) ;
					float zJitter = (random0to1() - 0.5f) ;
					float3 jitter = make_float3(xJitter, yJitter, zJitter) * particleSpacing * 0.5;

					float3 particlePos = subcellCenter;
					particlePos += jitter;

					float minDistanceFromWall = particleSpacing / 2;

					particlePos.x = max(0.0 + minDistanceFromWall, min(sizeX * cellPhysicalSize - minDistanceFromWall, particlePos.x));
					particlePos.y = max(0.0 + minDistanceFromWall, min(sizeY * cellPhysicalSize - minDistanceFromWall, particlePos.y));
					particlePos.z = max(0.0 + minDistanceFromWall, min(sizeZ * cellPhysicalSize - minDistanceFromWall, particlePos.z));

					if (filter(particlePos)) {
						particlesHost.emplace_back(particlePos);
						++createdCount;
					}
					
				}
			}
		}
		return createdCount;
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

	void Fluid::createSquareFluid(std::vector <Particle>& particlesHost, float3 minPos, float3 maxPos, int startIndex) {
		int index = startIndex;

		if (!checkCoordValid(minPos) || !checkCoordValid(maxPos)) {
			return;
		}


		for (int z =  minPos.z* sizeZ; z < maxPos.z * sizeZ; ++z) {
			for (int x = minPos.x * sizeX; x < maxPos.x * sizeX; ++x) {
				for (int y = minPos.y * sizeY; y < maxPos.y * sizeY; ++y) {

					++index;
					float3 thisPos = MAC_Grid_3D::getPhysicalPos(x, y, z, cellPhysicalSize);
					createParticlesAt(particlesHost, thisPos, [](float3 p) {return true; },cellPhysicalSize/2);
				}
			}
		}

		grid->fluidCount = index;
	}

	void Fluid::createSphereFluid(std::vector <Particle>& particlesHost,  float3 center, float radius,int startIndex) {
		int index = startIndex;
		if (!checkCoordValid(center)) {
			return;
		}
		float3 centerPos;
		centerPos.x = center.x * sizeX * cellPhysicalSize;
		centerPos.y = center.y * sizeY * cellPhysicalSize;
		centerPos.z = center.z * sizeZ * cellPhysicalSize;

		std::function <bool(float3)> filter = [&](float3 pos) {
			return length(pos - centerPos) < radius * cellPhysicalSize * sizeY;
		};

		for (int y = 0 * sizeY; y < 1 * sizeY; ++y) {
			for (int x = 0 * sizeX; x < 1 * sizeX; ++x) {
				for (int z = 0 * sizeZ; z < 1 * sizeZ; ++z) {
					float3 thisPos = MAC_Grid_3D::getPhysicalPos(x, y, z, cellPhysicalSize);
					int createdCount = createParticlesAt(particlesHost, thisPos, filter, cellPhysicalSize / 2);

					if (createdCount > 0) {
						
						++index;

					}
				}
			}
		}

		grid->fluidCount = index;
	}













	void Fluid::createSquareInk(std::vector <Particle>& particlesHost, float3 minPos, float3 maxPos, float spacing) {
		
		float minDistanceFromWall = spacing / 2.f;


		float3 gridPhysicalSize = make_float3(sizeX, sizeY, sizeZ) * cellPhysicalSize;

		float3 minPhysicalPos = {
			minPos.x * gridPhysicalSize.x,
			minPos.y * gridPhysicalSize.y,
			minPos.z * gridPhysicalSize.z,
		};
		minPhysicalPos += make_float3(1, 1, 1) * minDistanceFromWall;
		float3 maxPhysicalPos = {
			maxPos.x * gridPhysicalSize.x,
			maxPos.y * gridPhysicalSize.y,
			maxPos.z * gridPhysicalSize.z,
		};
		maxPhysicalPos -= make_float3(1, 1, 1) * (minDistanceFromWall - 1e-3);
		for (float x = minPhysicalPos.x; x <= maxPhysicalPos.x; x += spacing) {
			for (float y = minPhysicalPos.y; y <= maxPhysicalPos.y; y += spacing) {
				for (float z = minPhysicalPos.z; z <= maxPhysicalPos.z; z += spacing) {
					float jitterMagnitude = spacing / 2.f;
					float3 jitter;
					jitter.x = (random0to1() - 0.5);
					jitter.y = (random0to1() - 0.5);
					jitter.z = (random0to1() - 0.5);
					jitter *= jitterMagnitude;
					float3 pos = make_float3(x, y, z);
					pos += jitter;

					pos.x = min(gridPhysicalSize.x - minDistanceFromWall, max(minDistanceFromWall, pos.x));
					pos.y = min(gridPhysicalSize.y - minDistanceFromWall, max(minDistanceFromWall, pos.y));
					pos.z = min(gridPhysicalSize.z - minDistanceFromWall, max(minDistanceFromWall, pos.z));

					particlesHost.emplace_back(pos);

				}
			}
		}

	}

	void Fluid::createSphereInk(std::vector <Particle>& particlesHost, float3 center, float radius, float spacing) {
		if (!checkCoordValid(center)) {
			return;
		}
		float3 centerPos;
		centerPos.x = center.x * sizeX * cellPhysicalSize;
		centerPos.y = center.y * sizeY * cellPhysicalSize;
		centerPos.z = center.z * sizeZ * cellPhysicalSize;

		std::function <bool(float3)> filter = [&](float3 pos) {
			return length(pos - centerPos) < radius * cellPhysicalSize * sizeY;
		};

		for (float z = 0; z < sizeZ*cellPhysicalSize; z += spacing) {
			for (float y = 0; y < sizeY * cellPhysicalSize; y += spacing) {
				for (float x = 0; x < sizeX * cellPhysicalSize; x += spacing) {

					float3 thisPos = make_float3(x, y, z);
					createParticlesAt(particlesHost, thisPos,filter, cellPhysicalSize / 2);
				}
			}
		}

	}

	void Fluid::initInkRenderer() {
		std::vector <Particle> inkParticlesHost;

		inkParticlesSpacing = cellPhysicalSize / 4;


		for (const InitializationVolume& vol : fluidConfig->initialVolumes) {
			if (vol.shapeType == ShapeType::Square && vol.phase == 1) {
				float3 minPos = make_float3(vol.params[0], vol.params[1], vol.params[2]);
				float3 maxPos = make_float3(vol.params[3], vol.params[4], vol.params[5]);
				createSquareInk(inkParticlesHost, minPos, maxPos, inkParticlesSpacing);
			}
			else if (vol.shapeType == ShapeType::Sphere && vol.phase == 1) {
				float3 center = make_float3(vol.params[0], vol.params[1], vol.params[2]);
				float radius = vol.params[3];
				createSphereInk(inkParticlesHost,  center, radius, inkParticlesSpacing);
			}
		}


		inkParticleCount = inkParticlesHost.size();

		std::cout << "ink particle count: " << inkParticleCount << std::endl;
		std::cout << "ink particle malloc size: " << inkParticleCount * sizeof(Particle) << std::endl;

		HANDLE_ERROR(cudaMalloc(&inkParticles, inkParticleCount * sizeof(Particle)));

		HANDLE_ERROR(cudaMemcpy(inkParticles, inkParticlesHost.data(), inkParticleCount * sizeof(Particle),
			cudaMemcpyHostToDevice));


		pointSpritesInk = std::make_shared<PointSprites>(inkParticleCount);

		numThreadsInkParticle = min(1024, inkParticleCount);
		numBlocksInkParticle = divUp(inkParticleCount, numThreadsInkParticle);
	}

	void Fluid::drawInk(const DrawCommand& drawCommand) {

		transferToInkParticles();
		moveInkParticles(timestep);

		updatePositionsVBO << <numBlocksInkParticle, numThreadsInkParticle >> > (inkParticles, pointSpritesInk->positionsDevice, inkParticleCount, pointSprites->stride);
		cudaDeviceSynchronize();

		pointSpritesInk->drawInk(drawCommand, inkParticlesSpacing*0.7);

	}

	void Fluid::transferToInkParticles() {
		transferToParticlesImpl << <numBlocksInkParticle, numThreadsInkParticle >> > ( grid->volumes, inkParticles, inkParticleCount, sizeX, sizeY, sizeZ, cellPhysicalSize, 1.f);
		cudaDeviceSynchronize();
		CHECK_CUDA_ERROR("transfer to particles");
	}

	void Fluid::moveInkParticles(float timeStep) {

		moveParticlesImpl << < numBlocksInkParticle, numThreadsInkParticle >> >
			( grid->volumes, inkParticles, inkParticleCount, sizeX, sizeY, sizeZ, cellPhysicalSize, timeStep);
		cudaDeviceSynchronize();
		CHECK_CUDA_ERROR("move particles");
		return;

	}

}