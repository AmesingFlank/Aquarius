#include "Fluid_3D_PCISPH.cuh"



namespace Fluid_3D_PCISPH {







	__global__ void computeExternalForcesImpl(Particle* particles, int particleCount,float3 gravity) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= particleCount) return;

		Particle& particle = particles[index];
		particle.acceleration = gravity;

	}

	__global__ void initPressureImpl(Particle* particles, int particleCount) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= particleCount) return;

		Particle& particle = particles[index];
		particle.pressure = 0;
		particle.pressureForces = make_float3(0, 0, 0);

	}


	__global__ void predictVelocityAndPositionImpl(Particle* particles, int particleCount, float timestep, bool setAsActual, float spacing, float3 gridPhysicalSize) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= particleCount) return;

		Particle& particle = particles[index];

		float3 acc = particle.acceleration + particle.pressureForces;
		float3 vel = particle.velosity + acc * timestep;
		float3 pos = particle.position + vel * timestep;

		float bounce = -0.0;

		float minDistanceFromWall = spacing / 2;

		if (pos.x < minDistanceFromWall) {
			pos.x = minDistanceFromWall;
			vel.x *= bounce;;
		}

		if (pos.x > gridPhysicalSize.x - minDistanceFromWall) {
			pos.x = gridPhysicalSize.x - minDistanceFromWall;
			vel.x *= bounce;;
		}

		if (pos.y < minDistanceFromWall) {
			pos.y = minDistanceFromWall;
			vel.y *= bounce;;
		}

		if (pos.y > gridPhysicalSize.y - minDistanceFromWall) {
			pos.y = gridPhysicalSize.y - minDistanceFromWall;
			vel.y *= bounce;;
		}

		if (pos.z < minDistanceFromWall) {
			pos.z = minDistanceFromWall;
			vel.z *= bounce;;
		}

		if (pos.z > gridPhysicalSize.z - minDistanceFromWall) {
			pos.z = gridPhysicalSize.z - minDistanceFromWall;
			vel.z *= bounce;;
		}

		if (setAsActual) {
			particle.position = pos;
			particle.velosity = vel;

		}
		else {
			particle.predictedPosition = pos;
			particle.predictedVelocity = vel;
		}

	}

	__global__ void predictDensityAndPressureImpl(Particle* particles,int particleCount, int* cellBegin, int* cellEnd, int3 gridSize, float kernelRadius,float kernelRadius2,float kernelRadius6,float kernelRadius9, bool setAsRest, float timestep) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= particleCount) return;


		Particle& particle = particles[index];

		float3 pos = particle.position;
		int3 thisCell;

		thisCell.x = pos.x / kernelRadius;
		thisCell.y = pos.y / kernelRadius;
		thisCell.z = pos.z / kernelRadius;

		float rho0 = particle.restDensity;

		float beta = timestep * timestep * 2 / (rho0 * rho0);

		float density = 0;

		float3 sumGradW = make_float3(0, 0, 0);
		float sumGradWDot = 0;


#pragma unroll
		for (int dx = -1; dx <= 1; ++dx) {
#pragma unroll
			for (int dy = -1; dy <= 1; ++dy) {
#pragma unroll
				for (int dz = -1; dz <= 1; ++dz) {
					int x = thisCell.x + dx;
					int y = thisCell.y + dy;
					int z = thisCell.z + dz;
					if (x < 0 || x >= gridSize.x || y < 0 || y >= gridSize.y || z < 0 || z >= gridSize.z) {
						continue;
					}
					int hash = x * gridSize.y * gridSize.z + y * gridSize.z + z;
					if (cellBegin[hash] == -1) {
						continue;
					}
					for (int j = cellBegin[hash]; j <= cellEnd[hash]; ++j) {
						Particle& that = particles[j];
						float3 posDiff = particle.predictedPosition - that.predictedPosition;
						float thisDensityContribution = poly6(posDiff, kernelRadius2,kernelRadius9);



						density += thisDensityContribution;


						float3 gradW = spikey_grad(posDiff, kernelRadius,kernelRadius6);
						sumGradW += gradW;
						sumGradWDot += dot(gradW, gradW);
					}
				}
			}
		}
		particle.density = density;

		if (setAsRest) {
			particle.restDensity = density;
			
		}


		float rhoError = density - rho0;
		float correctionCoeff = 1.0 / (beta * (dot(sumGradW, sumGradW) + sumGradWDot));

		correctionCoeff = 50.0;

		float pressureCorrection = correctionCoeff * rhoError;
		particle.pressure += pressureCorrection;


	}

	__global__ void computePressureForceImpl(Particle* particles, int particleCount, int* cellBegin, int* cellEnd, int3 gridSize, float kernelRadius,float kernelRadius2, float kernelRadius6, float kernelRadius9 ) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= particleCount) return;


		Particle& particle = particles[index];

		float3 pos = particle.position;
		int3 thisCell;

		thisCell.x = pos.x / kernelRadius;
		thisCell.y = pos.y / kernelRadius;
		thisCell.z = pos.z / kernelRadius;

		float3 force = make_float3(0, 0, 0);

#pragma unroll
		for (int dx = -1; dx <= 1; ++dx) {
#pragma unroll
			for (int dy = -1; dy <= 1; ++dy) {
#pragma unroll
				for (int dz = -1; dz <= 1; ++dz) {
					int x = thisCell.x + dx;
					int y = thisCell.y + dy;
					int z = thisCell.z + dz;
					if (x < 0 || x >= gridSize.x || y < 0 || y >= gridSize.y || z < 0 || z >= gridSize.z) {
						continue;
					}
					int hash = x * gridSize.y * gridSize.z + y * gridSize.z + z;
					if (cellBegin[hash] == -1) {
						continue;
					}
					for (int j = cellBegin[hash]; j <= cellEnd[hash]; ++j) {
						Particle that = particles[j];
						force -= spikey_grad(particle.predictedPosition - that.predictedPosition, kernelRadius,kernelRadius6)
							* ((that.pressure / (that.density * that.density)) + (particle.pressure / (particle.density * particle.density)));
					}
				}
			}
		}
		particle.pressureForces = force;

	}







	Fluid::Fluid() {

	}

	void Fluid::draw(const DrawCommand& drawCommand){

		

		if (isMeshMode(drawCommand.renderMode)) {
			cudaDeviceSynchronize();
			mesher->mesh(particles, particlesCopy, particleHashes, particleIndices, meshRenderer->coordsDevice);
			cudaDeviceSynchronize();
			meshRenderer->draw(drawCommand);
		}
		else {
			updatePositionsVBO << <numBlocks, numThreads >> > (particles, pointSprites->positionsDevice, particleCount, pointSprites->stride);
			cudaDeviceSynchronize();
			pointSprites->draw(drawCommand, particleSpacing/2, drawCommand.texSkybox);
		}


	}

	void Fluid::createSquareFluid(std::vector<Particle>& particlesVec, float3 minPos, float3 maxPos) {
		float minDistanceFromWall = particleSpacing / 2.f;

		float3 minPhysicalPos = {
			minPos.x * gridPhysicalSize.x,
			minPos.y* gridPhysicalSize.y,
			minPos.z* gridPhysicalSize.z,
		};
		minPhysicalPos += make_float3(1, 1, 1) * minDistanceFromWall;
		float3 maxPhysicalPos = {
			maxPos.x* gridPhysicalSize.x,
			maxPos.y* gridPhysicalSize.y,
			maxPos.z* gridPhysicalSize.z,
		};
		maxPhysicalPos -= make_float3(1, 1, 1) * (minDistanceFromWall - 1e-3);
		for (float x = minPhysicalPos.x ; x <= maxPhysicalPos.x; x += particleSpacing) {
			for (float y = minPhysicalPos.y; y <= maxPhysicalPos.y ; y += particleSpacing) {
				for (float z = minPhysicalPos.z; z <= maxPhysicalPos.z ; z += particleSpacing) {
					float jitterMagnitude = particleSpacing/2.f;
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

					particlesVec.emplace_back(pos);

				}
			}
		}
	}
	void Fluid::createSphereFluid(std::vector<Particle>& particlesVec, float3 center, float radius) {

		float3 minPhysicalPos = {
			0,0,0
		};
		minPhysicalPos += make_float3(1, 1, 1) * particleSpacing * 0.5;
		float3 maxPhysicalPos = gridPhysicalSize;
		maxPhysicalPos -= make_float3(1, 1, 1) * particleSpacing * 0.5;

		float3 physicalCenter = {
			center.x * gridPhysicalSize.x,
			center.y * gridPhysicalSize.y,
			center.z * gridPhysicalSize.z
		};

		float physicalRadius = radius * gridPhysicalSize.y;

		for (float x = minPhysicalPos.x; x < maxPhysicalPos.x; x += particleSpacing) {
			for (float y = minPhysicalPos.y; y < maxPhysicalPos.y; y += particleSpacing) {
				for (float z = minPhysicalPos.z; z < maxPhysicalPos.z; z += particleSpacing) {

					float3 pos = make_float3(x, y, z);
					float3 jitter = make_float3(1, 1, 1);
					jitter.x *= (random0to1() - 0.5)*particleSpacing*0.01;
					jitter.y *= (random0to1() - 0.5) * particleSpacing * 0.01;
					jitter.z *= (random0to1() - 0.5) * particleSpacing * 0.01;


					
					if (length(pos-physicalCenter) < physicalRadius) {
						
						particlesVec.emplace_back(pos);
					}
				}
			}
		}
	}

	void Fluid::init(FluidConfig config) {

		particleCountWhenFull = config.PCISPH.maxParticleCount;
		stiffness = config.PCISPH.stiffness;
		timestep = config.PCISPH.timestep;
		substeps = config.PCISPH.substeps;
		iterations = config.PCISPH.iterations;

		this->fluidConfig = config;



		particleSpacing = pow(gridPhysicalSize.x * gridPhysicalSize.y * gridPhysicalSize.z / particleCountWhenFull, 1.0 / 3.0);

		particleSpacing = gridPhysicalSize.x / ceil(gridPhysicalSize.x / particleSpacing); // so that gridPhysicalSize is exact multiple.

		kernelRadius = particleSpacing * kernelRadiusToSpacingRatio;
		kernelRadius2 = kernelRadius * kernelRadius;
		kernelRadius6 = kernelRadius2 * kernelRadius2 * kernelRadius2;
		kernelRadius9 = kernelRadius6 * kernelRadius2 * kernelRadius;


		

		std::vector<Particle> particlesVec;

		for (const InitializationVolume& vol : config.initialVolumes) {
			if (vol.shapeType == ShapeType::Square) {
				createSquareFluid(particlesVec, vol.boxMin, vol.boxMax);
			}
			else if (vol.shapeType == ShapeType::Sphere) {

				createSphereFluid(particlesVec, vol.ballCenter, vol.ballRadius);
			}
		}

		
		particleCount = particlesVec.size();
		HANDLE_ERROR(cudaMalloc(&particles, particleCount * sizeof(Particle)));
		HANDLE_ERROR(cudaMalloc(&particlesCopy, particleCount * sizeof(Particle)));

		HANDLE_ERROR(cudaMemcpy(particles, particlesVec.data(), particleCount * sizeof(Particle), cudaMemcpyHostToDevice));

		numThreads = min(1024, particleCount);
		numBlocks = divUp(particleCount, numThreads);

		gridSize.x = ceil(gridPhysicalSize.x / kernelRadius);
		gridSize.y = ceil(gridPhysicalSize.y / kernelRadius);
		gridSize.z = ceil(gridPhysicalSize.z / kernelRadius);

		cellCount = gridSize.x * gridSize.y * gridSize.z;

		HANDLE_ERROR(cudaMalloc(&particleIndices, particleCount * sizeof(*particleIndices)));

		HANDLE_ERROR(cudaMalloc(&particleHashes, particleCount * sizeof(*particleHashes)));
		HANDLE_ERROR(cudaMalloc(&cellBegin, cellCount * sizeof(*cellBegin)));
		HANDLE_ERROR(cudaMalloc(&cellEnd, cellCount * sizeof(*cellEnd)));



		pointSprites = std::make_shared<PointSprites>(particleCount);


		computeRestDensity();
		HANDLE_ERROR(cudaMemcpy(particlesVec.data(), particles, particleCount * sizeof(Particle), cudaMemcpyDeviceToHost));

		float totalRestDensity = 0;
		float maxDensity = 0;
		float minDensity = 99999;
		for (Particle& p : particlesVec) {
			totalRestDensity += p.density;
			maxDensity = max(maxDensity, p.density);
			minDensity = min(minDensity, p.density);
		}
		restDensity = totalRestDensity / (float)particleCount;


		float variance = 0;
		for (Particle& p : particlesVec) {
			variance += pow(p.density - restDensity, 2);
		}
		variance /= (float)particleCount;

		std::cout << "particle count : " << particleCount << std::endl;


		std::cout << "spacing : " << particleSpacing << std::endl;
		std::cout << "kernel radius : " << kernelRadius << std::endl;
		std::cout << "rho0 : " << restDensity << std::endl;

		std::cout << "variance : " << variance << std::endl;
		std::cout << "gridSize.x : " << gridSize.x << std::endl;



		mesher = std::make_shared<Mesher>(gridPhysicalSize, particleSpacing, particleCount, numBlocks, numThreads);
		meshRenderer = std::make_shared<FluidMeshRenderer>(mesher->triangleCount);

		mesher->mesh(particles, particlesCopy, particleHashes, particleIndices, meshRenderer->coordsDevice);
		cudaDeviceSynchronize();

	}

	void Fluid::computeRestDensity() {
		performSpatialHashing2(particleIndices, particleHashes, particles, particlesCopy, particleCount, kernelRadius, gridSize.x, gridSize.y, gridSize.z, numBlocks, numThreads, cellBegin, cellEnd, cellCount);

		predictDensityAndPressureImpl << <numBlocks, numThreads >> >
			(particles,particleCount, cellBegin, cellEnd, gridSize, kernelRadius, kernelRadius2, kernelRadius6, kernelRadius9, true, timestep / (float)substeps);

	}



	void Fluid::simulationStep() {


		for (int i = 0; i < substeps; ++i) {

			performSpatialHashing2(particleIndices,particleHashes, particles, particlesCopy, particleCount, kernelRadius, gridSize.x, gridSize.y, gridSize.z, numBlocks, numThreads, cellBegin, cellEnd, cellCount);

			computeExternalForces();
			initPressure();

			int iter = 0;
			while (iter < iterations) {
				
				predictVelocityAndPosition();

				predictDensityAndPressure();

				computePressureForce();

				iter += 1;
			}

			computeNewVelocityAndPosition();
		}
		physicalTime += timestep;
	}

	void Fluid::computeExternalForces() {
		computeExternalForcesImpl << <numBlocks, numThreads >> > (particles, particleCount,fluidConfig.gravity);
	}

	void Fluid::initPressure() {
		initPressureImpl << <numBlocks, numThreads >> > (particles, particleCount);
	}


	void Fluid::predictVelocityAndPosition() {
		predictVelocityAndPositionImpl << <numBlocks, numThreads >> >
			(particles, particleCount, timestep / (float)substeps, false, particleSpacing, gridPhysicalSize);
	}

	void Fluid::predictDensityAndPressure() {
		predictDensityAndPressureImpl << <numBlocks, numThreads >> >
			(particles, particleCount, cellBegin, cellEnd, gridSize, kernelRadius, kernelRadius2, kernelRadius6, kernelRadius9, false, timestep / (float)substeps);
	}

	void Fluid::computePressureForce() {
		computePressureForceImpl << <numBlocks, numThreads >> >
			(particles,  particleCount, cellBegin, cellEnd, gridSize, kernelRadius, kernelRadius2, kernelRadius6, kernelRadius9 );
	}

	void Fluid::computeNewVelocityAndPosition() {
		predictVelocityAndPositionImpl << <numBlocks, numThreads >> >
			(particles, particleCount, timestep / (float)substeps, true, particleSpacing, gridPhysicalSize);
	}

	glm::vec3 Fluid::getCenter() {
		return glm::vec3(gridPhysicalSize.x / 2, gridPhysicalSize.y / 2,gridPhysicalSize.z / 2);
	}
	float Fluid::getContainerSize() {
		return gridPhysicalSize.x;
	}
	Fluid::~Fluid() {
		HANDLE_ERROR(cudaFree(particles));

		HANDLE_ERROR(cudaFree(particleHashes));
		HANDLE_ERROR(cudaFree(cellBegin));
		HANDLE_ERROR(cudaFree(cellEnd));



		HANDLE_ERROR(cudaFree(particleIndices));
		HANDLE_ERROR(cudaFree(particlesCopy));
	}
}