#pragma once

#ifndef FLUID_3D_PCISPH
#define FLUID_3D_PCISPH

#include "GpuCommons.h"
#include <vector>
#include <algorithm>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include "WeightKernels.h"
#include "../Rendering/Renderer3D/PointSprites.h"
#include "../Rendering/Renderer3D/Container.h"
#include "../Rendering/Renderer3D/Skybox.h"
#include "Fluid_3D.cuh"
#include "Fluid_3D_common.cuh"

namespace Fluid_3D_PCISPH {

	__host__ __device__ struct Particle {
		float3 position;
		float3 velosity = make_float3(0, 0, 0);
		float density;
		float pressure = 0;
		float restDensity;

		float3 pressureForces = make_float3(0, 0, 0);
		float3 acceleration = make_float3(0, 0, 0);

		float3 predictedPosition;
		float3 predictedVelocity = make_float3(0, 0, 0);

		float mass = 1;

		__host__ __device__ Particle(float3 position_) :position(position_),predictedPosition(position_) {

		}
		
		__host__ __device__ Particle() {

		}
	};
	


	__global__ void computeExternalForcesImpl(Particle* particles, int particleCount) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= particleCount) return;

		Particle& particle = particles[index];
		particle.acceleration = make_float3(0, -9.8, 0);

	}

	__global__ void initPressureImpl(Particle* particles, int particleCount) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= particleCount) return;

		Particle& particle = particles[index];
		particle.pressure = 0;
		particle.pressureForces = make_float3(0, 0, 0);

	}

	
	__global__ void predictVelocityAndPositionImpl(Particle* particles, int particleCount,float timestep,bool setAsActual,float spacing,float3 gridDimension) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= particleCount) return;

		Particle& particle = particles[index];
		
		float3 acc = particle.acceleration + particle.pressureForces;
		float3 vel = particle.velosity + acc * timestep;
		float3 pos = particle.position + vel * timestep;

		float bounce = -0.0;

		if (pos.x < spacing) {
			pos.x = spacing;
			vel.x *= bounce;;
		}

		if (pos.x > gridDimension.x - spacing) {
			pos.x = gridDimension.x - spacing;
			vel.x *= bounce;;
		}

		if (pos.y < spacing) {
			pos.y = spacing;
			vel.y *= bounce;;
		}

		if (pos.y > gridDimension.y - spacing) {
			pos.y = gridDimension.y - spacing;
			vel.y *= bounce;;
		}

		if (pos.z < spacing) {
			pos.z = spacing;
			vel.z *= bounce;;
		}

		if (pos.z > gridDimension.z - spacing) {
			pos.z = gridDimension.z - spacing;
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

	__global__ void predictDensityAndPressureImpl(Particle* particles, float cellSize, int particleCount, int* cellBegin, int* cellEnd, int3 gridSize, float kernelRadius, bool setAsRest,float timestep) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= particleCount) return;


		Particle& particle = particles[index];

		float3 pos = particle.position;
		int3 thisCell;

		thisCell.x = pos.x / cellSize;
		thisCell.y = pos.y / cellSize;
		thisCell.z = pos.z / cellSize;

		float rho0 = particle.restDensity ;

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
						density += poly6(posDiff, kernelRadius);
						float3 gradW = spikey_grad(posDiff, kernelRadius);
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

		if (index == 666) {
			//printf("rho: %f \n", density);
			//printf("stiff: %f \n", correctionCoeff);
		}
	}

	__global__ void computePressureForceImpl(Particle* particles, float cellSize, int particleCount, int* cellBegin, int* cellEnd, int3 gridSize, float kernelRadius) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= particleCount) return;


		Particle& particle = particles[index];

		float3 pos = particle.position;
		int3 thisCell;

		thisCell.x = pos.x / cellSize;
		thisCell.y = pos.y / cellSize;
		thisCell.z = pos.z / cellSize;

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
						force -= spikey_grad(particle.predictedPosition - that.predictedPosition, kernelRadius)
							* ((that.pressure / (that.density*that.density)) + (particle.pressure / (particle.density*particle.density))) ;
					}
				}
			}
		}
		particle.pressureForces = force;

	}


	class Fluid : public Fluid_3D{
	public:
		int particleCount;
		int cellCount;

		Particle* particles;
		Particle* particlesCopy;
		int* particleIndices;
		int* particleHashes;
		int* cellBegin;
		int* cellEnd;

		float timestep = 5e-3;
		float substeps = 1;

		float3 gridDimension = make_float3(10.f, 10.f, 10.f);

		int3 gridSize;

		float restDensity;

		float particleCountWhenFull = 3e5;

		float kernelRadiusToSpacingRatio = 3.5;

		float stiffness = 15;

		float kernelRadius;

		float particleSpacing;

		float minIterations = 4;

		float maxIterations = 10;


		PointSprites* pointSprites;

		int numThreads, numBlocks;

		Container container = Container(glm::vec3(gridDimension.x, gridDimension.y, gridDimension.z));

		Skybox skybox = Skybox("resources/Park2/",".jpg");

		std::shared_ptr<Mesher> mesher;
		std::shared_ptr<FluidMeshRenderer> meshRenderer;


		Fluid() {

			initFluid();
		}



		void initFluid() {
			particleSpacing = pow(gridDimension.x * gridDimension.y * gridDimension.z / particleCountWhenFull, 1.0 / 3.0);

			kernelRadius = particleSpacing * kernelRadiusToSpacingRatio;


			std::vector<Particle> particlesVec;
			for (float x = particleSpacing; x < gridDimension.x - particleSpacing; x += particleSpacing) {
				for (float y = particleSpacing; y < gridDimension.y - particleSpacing; y += particleSpacing) {
					for (float z = particleSpacing; z < gridDimension.z - particleSpacing; z += particleSpacing) {
						if (x <= gridDimension.x/2 && y <= gridDimension.y / 2 && z <= gridDimension.z/2) {
							particlesVec.emplace_back(make_float3(x, y, z));
						}
						else if (length(make_float3(x - gridDimension.x * 0.5, y - gridDimension.y * 0.75, z - gridDimension.z * 0.5)) < gridDimension.x * 0.2) {
							//particlesVec.emplace_back(make_float3(x, y, z));
						}
					}
				}
			}
			particleCount = particlesVec.size();
			HANDLE_ERROR(cudaMalloc(&particles, particleCount * sizeof(Particle)));
			HANDLE_ERROR(cudaMalloc(&particlesCopy, particleCount * sizeof(Particle)));

			HANDLE_ERROR(cudaMemcpy(particles, particlesVec.data(), particleCount * sizeof(Particle), cudaMemcpyHostToDevice));

			numThreads = min(1024, particleCount);
			numBlocks = divUp(particleCount, numThreads);

			gridSize.x = ceil(gridDimension.x / kernelRadius);
			gridSize.y = ceil(gridDimension.y / kernelRadius);
			gridSize.z = ceil(gridDimension.z / kernelRadius);

			cellCount = gridSize.x * gridSize.y * gridSize.z;

			HANDLE_ERROR(cudaMalloc(&particleIndices, particleCount * sizeof(*particleIndices)));

			HANDLE_ERROR(cudaMalloc(&particleHashes, particleCount * sizeof(*particleHashes)));
			HANDLE_ERROR(cudaMalloc(&cellBegin, cellCount * sizeof(*cellBegin)));
			HANDLE_ERROR(cudaMalloc(&cellEnd, cellCount * sizeof(*cellEnd)));



			pointSprites = new PointSprites(particleCount);

			std::cout << "particle count : " << particleCount << std::endl;

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


			std::cout << "spacing : " << particleSpacing << std::endl;
			std::cout << "kernel radius : " << kernelRadius << std::endl;
			std::cout << "rho0 : " << restDensity << std::endl;

			std::cout << "variance : " << variance << std::endl;


			mesher = std::make_shared<Mesher>(gridSize.x, gridSize.y, gridSize.z, particleCount, numBlocks, numThreads);
			meshRenderer = std::make_shared<FluidMeshRenderer>(mesher->triangleCount);



		}

		void updateVBO() {
			updatePositionsVBO << <numBlocks, numThreads >> > (particles, pointSprites->positionsDevice, particleCount);
		}



		virtual void draw(const DrawCommand& drawCommand) override {
			skybox.draw(drawCommand);
			updateVBO();
			container.draw(drawCommand);

			if (drawCommand.renderMode == RenderMode::Mesh) {
				mesher->mesh(particles, particlesCopy, particleHashes, particleIndices, meshRenderer->coordsDevice, gridDimension);
				cudaDeviceSynchronize();
				meshRenderer->draw(drawCommand);
			}
			else {
				pointSprites->draw(drawCommand, particleSpacing, skybox.texSkyBox);
			}

		}

		virtual void init(std::shared_ptr<FluidConfig> config) {

		}

		void computeRestDensity() {
			performSpatialHashing2(particleIndices, particleHashes, particles, particlesCopy, particleCount, kernelRadius, gridSize.x, gridSize.y, gridSize.z, numBlocks, numThreads, cellBegin, cellEnd,cellCount);

			predictDensityAndPressureImpl << <numBlocks, numThreads >> >
			(particles, kernelRadius, particleCount, cellBegin, cellEnd, gridSize, kernelRadius, true, timestep / (float)substeps);

		}


		virtual void simulationStep() override {
			for (int i = 0; i < substeps; ++i) {

				performSpatialHashing(particleHashes, particles, particleCount, kernelRadius, gridSize.x, gridSize.y, gridSize.z, numBlocks, numThreads, cellBegin, cellEnd, cellCount);
				computeExternalForces();
				initPressure();

				int iter = 0;
				while (iter < minIterations || hasBigError()) {
					if (iter > maxIterations) {
						//std::cout << "hit max iters" << std::endl;
						break;
					}
					predictVelocityAndPosition();
					
					predictDensityAndPressure();

					computePressureForce();

					iter += 1;
				}

				computeNewVelocityAndPosition();
			}
		}

		void computeExternalForces() {
			computeExternalForcesImpl << <numBlocks, numThreads >> > (particles, particleCount);
		}

		void initPressure() {
			initPressureImpl << <numBlocks, numThreads >> > (particles, particleCount);
		}

		bool hasBigError() {
			return true;
		}

		void predictVelocityAndPosition() {
			predictVelocityAndPositionImpl << <numBlocks, numThreads >> > 
				(particles, particleCount, timestep / (float)substeps,false,particleSpacing,gridDimension);
		}

		void predictDensityAndPressure() {
			predictDensityAndPressureImpl << <numBlocks, numThreads >> >
			(particles, kernelRadius, particleCount, cellBegin, cellEnd, gridSize, kernelRadius, false, timestep / (float)substeps);
		}

		void computePressureForce() {
			computePressureForceImpl << <numBlocks, numThreads >> >
				(particles, kernelRadius, particleCount, cellBegin, cellEnd, gridSize, kernelRadius);
		}

		void computeNewVelocityAndPosition() {
			predictVelocityAndPositionImpl << <numBlocks, numThreads >> >
				(particles, particleCount, timestep / (float)substeps, true, particleSpacing, gridDimension);
		}



	};
}

#endif