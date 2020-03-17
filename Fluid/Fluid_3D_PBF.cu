#include "Fluid_3D_PBF.cuh"

namespace Fluid_3D_PBF {


	__global__ void applyForcesImpl(Particle* particles, int particleCount, float timestep,float3 gravity) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= particleCount) return;

		Particle& particle = particles[index];
		particle.velosity += timestep * gravity;
		
	}

	__global__ void predictPositionImpl(Particle* particles, int particleCount, float timestep,float3 gridPhysicalSize,float minDistanceFromWall) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= particleCount) return;


		Particle& particle = particles[index];
		particle.position = particle.lastPosition + timestep * particle.velosity;

		particle.position.x = min(gridPhysicalSize.x - minDistanceFromWall, max(minDistanceFromWall, particle.position.x));
		particle.position.y = min(gridPhysicalSize.y - minDistanceFromWall, max(minDistanceFromWall, particle.position.y));
		particle.position.z = min(gridPhysicalSize.z - minDistanceFromWall, max(minDistanceFromWall, particle.position.z));
	}


	__global__ void computeDensityAndLambdaImpl(Particle* particles, int particleCount, int* cellBegin, int* cellEnd, int3 gridSize, float kernelRadius,float kernelRadius2,float kernelRadius3,float kernelRadius5,float kernelRadius6,float kernelRadius9,bool setDensityAsRestDensity) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= particleCount) return;


		Particle& particle = particles[index];

		float3 pos = particle.position;
		int3 thisCell;

		thisCell.x = pos.x / kernelRadius;
		thisCell.y = pos.y / kernelRadius;
		thisCell.z = pos.z / kernelRadius;



		float3 grad_pi_Ci = make_float3(0,0, 0);
		float sum_dot_grad_pj_Ci = 0;

		float density = 0;


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
						float3 posDiff = particle.position - that.position;

						
						density += poly6(posDiff, kernelRadius2,kernelRadius9);
						float3 gradientWeight = spikey_grad(posDiff, kernelRadius,kernelRadius6);
						
						
						/*
						float posDiffLength = length(posDiff);
						float densityWeight = cubic_spline_kernel(posDiffLength, kernelRadius,kernelRadius3);
						density += densityWeight;
						float3 gradientWeight = cubic_spline_kernel_gradient(posDiff, posDiffLength,kernelRadius,kernelRadius5);
						*/

						grad_pi_Ci += gradientWeight;
						sum_dot_grad_pj_Ci += dot(gradientWeight, gradientWeight);

						
					}
				}
			}
		}

		particle.density = density;

		if (setDensityAsRestDensity) {
			particle.restDensity = density;
		}

		float Ci = particle.density / particle.restDensity - 1;



		float denominator = 1e-3 + (sum_dot_grad_pj_Ci + dot(grad_pi_Ci, grad_pi_Ci)) / (particle.restDensity*particle.restDensity);

		

		float lambda = -Ci / denominator;

		particle.lambda = lambda;


		if (density != particle.restDensity) {
			//printf("density:%f   rho0:%f   ci:%f   denom:%f   lambda:%f   \n", density,particle.restDensity,Ci, denominator,lambda);
		}

		if (Ci < 0) {
			particle.lambda *= 1e-2;
		}

	}



	__global__ void computeDeltaPositionImpl(Particle* particles, int particleCount, int* cellBegin, int* cellEnd, int3 gridSize, float kernelRadius,float kernelRadius2, float kernelRadius3, float kernelRadius5, float kernelRadius6, float kernelRadius9, float s_corr_k, float s_corr_denominator) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= particleCount) return;


		Particle& particle = particles[index];

		float3 pos = particle.position;
		int3 thisCell;


		thisCell.x = pos.x / kernelRadius;
		thisCell.y = pos.y / kernelRadius;
		thisCell.z = pos.z / kernelRadius;

		float3 deltaPosition = make_float3(0, 0, 0);



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
						float3 posDiff = particle.position - that.position;
						

						float posDiffLength = length(posDiff);

						float3 gradientWeight = spikey_grad(posDiff,  kernelRadius,kernelRadius6);
						float densityWeight = poly6(posDiff, kernelRadius2, kernelRadius9);
						
						//float3 gradientWeight = cubic_spline_kernel_gradient(posDiff, posDiffLength, kernelRadius,kernelRadius5);
						//float densityWeight = cubic_spline_kernel(posDiffLength, kernelRadius, kernelRadius3);

						float s_corr_temp = densityWeight / s_corr_denominator;
						float s_corr_temp_2 = s_corr_temp * s_corr_temp;
						float s_corr_temp_4 = s_corr_temp_2 * s_corr_temp_2;
						float s_corr = s_corr_temp_4 * (-s_corr_k);

						deltaPosition += gradientWeight * (particle.lambda + that.lambda+s_corr ) / particle.restDensity;

						
					}
				}
			}
		}

		particle.deltaPosition = deltaPosition;


	}

	__global__ void applyDeltaPositionImpl(Particle* particles, int particleCount, float3 gridPhysicalSize,float minDistanceFromWall) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= particleCount) return;


		Particle& particle = particles[index];
		float3 deltaPosition = particle.deltaPosition;


		particle.position += deltaPosition;

		particle.position.x = min(gridPhysicalSize.x - minDistanceFromWall, max(minDistanceFromWall, particle.position.x));
		particle.position.y = min(gridPhysicalSize.y - minDistanceFromWall, max(minDistanceFromWall, particle.position.y));
		particle.position.z = min(gridPhysicalSize.z - minDistanceFromWall, max(minDistanceFromWall, particle.position.z));


	}


	__global__ void updatePositionAndVelocityImpl(Particle* particles, int particleCount, float timestep,float3 gridPhysicalSize,float minDistanceFromWall) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= particleCount) return;


		Particle& particle = particles[index];

		particle.velosity = (particle.position - particle.lastPosition) / timestep;
		particle.lastPosition = particle.position;

		float3 pos = particle.position;
		float3& vel = particle.velosity;

		float leaveWall = minDistanceFromWall / (timestep * 1e2);

		float bounce = -0.5;

		if (pos.x == minDistanceFromWall) {
			if (vel.x == 0) 
				vel.x = leaveWall;
			else 
				vel.x *= bounce;
		}

		if (pos.x >= gridPhysicalSize.x - minDistanceFromWall) {
			if (vel.x == 0) 
				vel.x = -leaveWall;
			else 
				vel.x *= bounce;;
		}

		if (pos.y <= minDistanceFromWall) {
			if (vel.y == 0)
				vel.y = leaveWall;
			else
				vel.y *= bounce;
		}

		if (pos.y >= gridPhysicalSize.y - minDistanceFromWall) {
			if (vel.y == 0)
				vel.y = -leaveWall;
			else
				vel.y *= bounce;;
		}

		if (pos.z <= minDistanceFromWall) {
			if (vel.z == 0)
				vel.z = leaveWall;
			else
				vel.z *= bounce;
		}

		if (pos.z >= gridPhysicalSize.z - minDistanceFromWall) {
			if (vel.z == 0)
				vel.z = -leaveWall;
			else
				vel.z *= bounce;;
		}
	}





	Fluid::Fluid() {

	}
	
	void Fluid::init(FluidConfig config) {
		
		substeps = config.PBF.substeps;
		timestep = config.PBF.timestep;
		solverIterations = config.PBF.iterations;
		particleCountWhenFull = config.PBF.maxParticleCount;



		particleSpacing = pow(gridPhysicalSize.x * gridPhysicalSize.y * gridPhysicalSize.z / particleCountWhenFull, 1.0 / 3.0);
		particleSpacing = gridPhysicalSize.x / ceil(gridPhysicalSize.x / particleSpacing); // so that gridPhysicalSize is exact multiple.

		//particleSpacing = 0.02f;



		kernelRadius = particleSpacing * kernelRadiusToSpacingRatio;
		kernelRadius2 = kernelRadius * kernelRadius;
		kernelRadius3 = kernelRadius2 * kernelRadius;
		kernelRadius5 = kernelRadius3 * kernelRadius2;
		kernelRadius6 = kernelRadius2 * kernelRadius2 * kernelRadius2;
		kernelRadius9 = kernelRadius6 * kernelRadius2 * kernelRadius;


		std::vector<Particle> particlesVec;


		fluidConfig = config;


		for (const InitializationVolume& vol : config.initialVolumes) {
			if (vol.shapeType == ShapeType::Square) {
				float3 minPos = make_float3(vol.params[0], vol.params[1], vol.params[2]);
				float3 maxPos = make_float3(vol.params[3], vol.params[4], vol.params[5]);
				createSquareFluid(particlesVec, minPos, maxPos);
			}
			else if (vol.shapeType == ShapeType::Sphere) {
				float3 center = make_float3(vol.params[0], vol.params[1], vol.params[2]);
				float radius = vol.params[3];
				createSphereFluid(particlesVec, center, radius);
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

		/*
		for (Particle& p : particlesVec) {
			p.restDensity = 1;
		}*/
		//HANDLE_ERROR(cudaMemcpy(particles, particlesVec.data(), particleCount * sizeof(Particle), cudaMemcpyHostToDevice));


		std::cout << "particle count : " << particleCount << std::endl;

		std::cout << "spacing : " << particleSpacing << std::endl;
		std::cout << "kernel radius : " << kernelRadius << std::endl;

		std::cout << "mean rho0 : " << restDensity << std::endl;
		std::cout << "min rho0 : " << minDensity << std::endl;
		std::cout << "max rho0 : " << maxDensity << std::endl;

		std::cout << "variance : " << variance << std::endl;
		std::cout << "gridSize.x : " << gridSize.x << std::endl;



		mesher = std::make_shared<Mesher>(gridPhysicalSize, particleSpacing, particleCount, numBlocks, numThreads);
		meshRenderer = std::make_shared<FluidMeshRenderer>(mesher->triangleCount);

		mesher->mesh(particles, particlesCopy, particleHashes, particleIndices, meshRenderer->coordsDevice);
		cudaDeviceSynchronize();

		container = std::make_shared<Container>(gridPhysicalSize.x);

	}

	void Fluid::createSquareFluid(std::vector<Particle>& particlesVec, float3 minPos, float3 maxPos) {
		float minDistanceFromWall = particleSpacing / 2.f;

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
		for (float x = minPhysicalPos.x; x <= maxPhysicalPos.x; x += particleSpacing) {
			for (float y = minPhysicalPos.y; y <= maxPhysicalPos.y; y += particleSpacing) {
				for (float z = minPhysicalPos.z; z <= maxPhysicalPos.z; z += particleSpacing) {
					float jitterMagnitude = particleSpacing / 2.f;
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
					jitter.x *= (random0to1() - 0.5) * particleSpacing * 0.5;
					jitter.y *= (random0to1() - 0.5) * particleSpacing * 0.5;
					jitter.z *= (random0to1() - 0.5) * particleSpacing * 0.5;

					pos += jitter;

					if (length(pos - physicalCenter) < physicalRadius) {

						particlesVec.emplace_back(pos);
					}
				}
			}
		}
	}

	void Fluid::computeRestDensity() {
		performSpatialHashing2(particleIndices, particleHashes, particles, particlesCopy, particleCount, kernelRadius, gridSize.x, gridSize.y, gridSize.z, numBlocks, numThreads, cellBegin, cellEnd, cellCount);

		computeDensityAndLambda(true);
	}




	void Fluid::simulationStep() {
		performSpatialHashing2(particleIndices, particleHashes, particles, particlesCopy, particleCount, kernelRadius, gridSize.x, gridSize.y, gridSize.z, numBlocks, numThreads, cellBegin, cellEnd, cellCount);



		for (int i = 0; i < substeps; ++i) {
			applyForces();
			predictPosition();

			performSpatialHashing2(particleIndices, particleHashes, particles, particlesCopy, particleCount, kernelRadius, gridSize.x, gridSize.y, gridSize.z, numBlocks, numThreads, cellBegin, cellEnd, cellCount);


			float t0 = glfwGetTime();
			for (int iter = 0; iter < solverIterations; ++iter) {
				computeDensityAndLambda(false);
				computeDeltaPosition();
				applyDeltaPosition();
			}
			float t1 = glfwGetTime();


			updateVelocityAndPosition();
			
		}

		physicalTime += timestep;
	}

	void Fluid::applyForces() {
		applyForcesImpl<<<numBlocks,numThreads>>>(particles, particleCount, timestep / (float)substeps,fluidConfig.gravity);
	}

	void Fluid::predictPosition() {
		predictPositionImpl << <numBlocks, numThreads >> > (particles, particleCount, timestep / (float)substeps,gridPhysicalSize,particleSpacing/2.f);
	}

	void Fluid::computeDensityAndLambda(bool setDensityAsRestDensity) {
		


		computeDensityAndLambdaImpl << <numBlocks, numThreads >> > (particles, particleCount, cellBegin, cellEnd, gridSize, kernelRadius,kernelRadius2, kernelRadius3, kernelRadius5, kernelRadius6, kernelRadius9,setDensityAsRestDensity);
		CHECK_CUDA_ERROR("lambda");

		
	}

	void Fluid::computeDeltaPosition() {

		float s_corr_k = 1e-6;
		float s_corr_delta_q = 0;
		float s_corr_denominator = cubic_spline_kernel(s_corr_delta_q, kernelRadius, kernelRadius3);

		computeDeltaPositionImpl << <numBlocks, numThreads >> >  (particles, particleCount, cellBegin, cellEnd, gridSize, kernelRadius, kernelRadius2, kernelRadius3, kernelRadius5, kernelRadius6, kernelRadius9,s_corr_k,s_corr_denominator);
		CHECK_CUDA_ERROR("deltapos");
	}

	void Fluid::applyDeltaPosition() {
		applyDeltaPositionImpl << <numBlocks, numThreads >> > (particles, particleCount, gridPhysicalSize, particleSpacing / 2.f);
	}

	void Fluid::updateVelocityAndPosition() {
		updatePositionAndVelocityImpl << <numBlocks, numThreads >> > (particles, particleCount, timestep / (float)substeps, gridPhysicalSize,particleSpacing / 2.f);
	}




	void Fluid::draw(const DrawCommand& drawCommand) {
		skybox.draw(drawCommand);
		container->draw(drawCommand);

		if (drawCommand.renderMode == RenderMode::Mesh) {
			if (!drawCommand.simulationPaused) {
				mesher->mesh(particles, particlesCopy, particleHashes, particleIndices, meshRenderer->coordsDevice);
			}
			cudaDeviceSynchronize();
			meshRenderer->draw(drawCommand, skybox.texSkyBox);
		}
		else {
			updatePositionsVBO << <numBlocks, numThreads >> > (particles, pointSprites->positionsDevice, particleCount,pointSprites->stride);
			CHECK_CUDA_ERROR("update positions and colors");
			cudaDeviceSynchronize();
			pointSprites->draw(drawCommand, particleSpacing / 2, skybox.texSkyBox);
		}
		
	}

	glm::vec3 Fluid::getCenter() {
		return glm::vec3(gridPhysicalSize.x / 2, gridPhysicalSize.y / 2, gridPhysicalSize.z / 2);
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