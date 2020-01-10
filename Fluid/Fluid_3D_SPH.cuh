#pragma once

#ifndef FLUID_3D_SPH
#define FLUID_3D_SPH

#include "GpuCommons.h"
#include <vector>
#include <algorithm>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include "WeightKernels.cuh"
#include "Rendering/Renderer3D/PointSprites.h"
#include "Rendering/Renderer3D/Container.h"

namespace Fluid_3D_SPH{

	__host__ __device__ struct Particle {
		float3 position;
		float3 velosity = make_float3(0, 0, 0);
		float density;
		float pressure;
		float restDensity;
		float3 acceleration = make_float3(0, 0, 0);
		float3 newAcceleration = make_float3(0, 0, 0);
		__host__ __device__ Particle(float3 position_) :position(position_) {

		}
		__host__ __device__ Particle() {

		}
	};
	__global__ void updatePositionsVBO(Particle* particles,float* positionsVBO,int particleCount) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= particleCount) return;

		float* base = positionsVBO + index * 3;
		Particle& particle = particles[index];


		base[0] = particle.position.x;
		base[1] = particle.position.y;
		base[2] = particle.position.z;
	}
	__global__ void calcHashImpl(Particle* particles, int* particleHashes, float cellSize,int particleCount,int3 gridSize) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= particleCount) return;

		Particle& particle = particles[index];
		float3 pos = particle.position;

		int x = pos.x / cellSize;
		int y = pos.y / cellSize;
		int z = pos.z / cellSize;

		int hash = x * gridSize.y * gridSize.z + y * gridSize.z + z;

		particleHashes[index] = hash;

	}

	__global__ void findCellStartEndImpl( int* particleHashes, int* cellBegin,int* cellEnd,int particleCount) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= particleCount) return;

		
		int hash = particleHashes[index];

		if (index == 0 || particleHashes[index - 1] < hash) {
			cellBegin[hash] = index;
		}

		if (index == particleCount - 1 || particleHashes[index + 1] > hash) {
			cellEnd[hash] = index;
		}

	}

	__global__ void updatePositionImpl(Particle* particles, int particleCount, float3 gridPhysicalSize,float spacing,float timeStep) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= particleCount) return;

		Particle& particle = particles[index];
		particle.position = particle.position + particle.velosity * timeStep + particle.acceleration * timeStep * timeStep / 2.0;

		float bounce = -0.0;

		if (particle.position.x < spacing) {
			particle.position.x = spacing;
			particle.velosity.x *= bounce;;
		}

		if (particle.position.x > gridPhysicalSize.x-spacing) {
			particle.position.x = gridPhysicalSize.x - spacing;
			particle.velosity.x *= bounce;;
		}

		if (particle.position.y < spacing) {
			particle.position.y = spacing;
			particle.velosity.y *= bounce;;
		}

		if (particle.position.y > gridPhysicalSize.y - spacing) {
			particle.position.y = gridPhysicalSize.y - spacing;
			particle.velosity.y *= bounce;;
		}

		if (particle.position.z < spacing) {
			particle.position.z = spacing;
			particle.velosity.z *= bounce;;
		}

		if (particle.position.z > gridPhysicalSize.z - spacing) {
			particle.position.z = gridPhysicalSize.z - spacing;
			particle.velosity.z *= bounce;;
		}
	}

	__global__ void addExternalForcesImpl(Particle* particles, int particleCount) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= particleCount) return;


		Particle& particle = particles[index];
		particle.newAcceleration = make_float3(0, -9.8, 0);

	}

	__global__ void calculateDensityImpl(Particle* particles,float cellSize, int particleCount,int*cellBegin,int* cellEnd,int3 gridSize,float kernelRadius,bool setAsRest) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= particleCount) return;


		Particle& particle = particles[index];

		float3 pos = particle.position;
		int3 thisCell;

		thisCell.x = pos.x / cellSize;
		thisCell.y = pos.y / cellSize;
		thisCell.z = pos.z / cellSize;

		float density = 0;

#pragma unroll
		for (int dx = -1; dx <=  1; ++dx) {
#pragma unroll
			for (int dy = -1; dy <= 1; ++dy) {
#pragma unroll
				for (int dz = - 1; dz <= 1; ++dz) {
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
						density += poly6(particles[j].position - pos, kernelRadius);
					}
				}
			}
		}
		particle.density = density;
		if (index == 1666) {
			//printf("density == %f \n", density);
		}
		if (setAsRest) {
			particle.restDensity = density;
		}
	}

	__global__ void calculatePressureImpl(Particle* particles, int particleCount,float restDensity,float stiffness) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= particleCount) return;


		Particle& particle = particles[index];
		particle.pressure = stiffness * (particle.density - particle.restDensity);

	}

	__global__ void addPressureForceImpl(Particle* particles, float cellSize, int particleCount, int* cellBegin, int* cellEnd, int3 gridSize, float kernelRadius) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= particleCount) return;


		Particle& particle = particles[index];

		float3 pos = particle.position;
		int3 thisCell;

		thisCell.x = pos.x / cellSize;
		thisCell.y = pos.y / cellSize;
		thisCell.z = pos.z / cellSize;

		float3 force = make_float3(0,0,0);

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
						force -= spikey_grad( pos-that.position, kernelRadius) * (that.pressure+particle.pressure) 
							/ (2.0*that.density);
					}
				}
			}
		}
		particle.newAcceleration += force;

	}

	__global__ void updateVelocityImpl(Particle* particles, int particleCount,float timeStep) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= particleCount) return;


		Particle& particle = particles[index];
		particle.velosity = particle.velosity + (particle.acceleration + particle.newAcceleration) * timeStep / 2.0;


		particle.acceleration = particle.newAcceleration;
	}

	class Fluid {
	public:
		int particleCount;
		int cellCount;

		Particle* particles;
		int* particleHashes;
		int* cellBegin;
		int* cellEnd;

		float timestep = 1e-2;
		float substeps = 10;

		float3 gridPhysicalSize = make_float3(10.f, 10.f, 10.f);

		int3 gridSize;

		float restDensity;

		float particleCountWhenFull = 3e4;

		float kernelRadiusToSpacingRatio = 3.5;

		float stiffness = 15;

		float kernelRadius;

		float particleSpacing;

		PointSprites* pointSprites;

		int numThreads, numBlocks;

		Container container = Container(glm::vec3(gridPhysicalSize.x,gridPhysicalSize.y,gridPhysicalSize.z));


		Fluid() {

			//computeGridSizes();
			//computeParticleSpacing();

			initFluid();
		}


		/*

		void computeGridSizes() {
			float l = 0;
			float r = particleRadius * 100;
			while (abs(r-l)/l > 1e-6) {
				float m = (l + r) / 2;
				float density = poly6(make_float3(0, 0, 0), m);
				if (density == SCP * restDensity) {
					break;
				}
				if (density > SCP* restDensity) {
					l = m;
				}
				else {
					r = m;
				}
			}
			kernelRadius = l;
			gridSize.x = ceil(gridDimension.x / kernelRadius);
			gridSize.y = ceil(gridDimension.y / kernelRadius);
			gridSize.z = ceil(gridDimension.z / kernelRadius);

			std::cout << "particle radius : " << particleRadius << std::endl;
			std::cout << "kernel radius : " << kernelRadius << std::endl;
			std::cout << "grid size x : " << gridSize.x << std::endl;

		}

		void computeParticleSpacing() {
			int neighboursPerSide = 4;
			int sides = 6;
			double l = 0;
			double r = kernelRadius;
			while (abs(r - l) / l > 1e-6) {
				double m = (r + l) / 2;
				float contributionPerSide = 0;
				for (int i = 1; i <= neighboursPerSide; ++i) {
					contributionPerSide += poly6(make_float2(m * i, 0), kernelRadius);
				}
				float density = poly6(make_float2(0, 0), kernelRadius) + contributionPerSide * sides;
				if (density == restDensity) {
					break;
				}
				if (density > restDensity) {
					l = m;
				}
				else {
					r = m;
				}
			}
			particleSpacing = (r + l) / 2;
			std::cout << "particle seperation : " << particleSpacing << std::endl;
			

		}
		*/

		void initFluid() {
			particleSpacing = pow(gridPhysicalSize.x * gridPhysicalSize.y * gridPhysicalSize.z / particleCountWhenFull, 1.0 / 3.0);

			kernelRadius = particleSpacing * kernelRadiusToSpacingRatio;

			std::vector<Particle> particlesVec;
			for (float x = particleSpacing; x < gridPhysicalSize.x - particleSpacing; x+=particleSpacing) {
				for (float y = particleSpacing; y < gridPhysicalSize.y - particleSpacing; y+=particleSpacing) {
					for (float z = particleSpacing; z < gridPhysicalSize.z - particleSpacing; z += particleSpacing) {
						if (x <= gridPhysicalSize.x  && y <= gridPhysicalSize.y/2  && z <= gridPhysicalSize.z ) {
							particlesVec.emplace_back(make_float3(x, y, z));
						}
						else if (length(make_float3(x-gridPhysicalSize.x*0.5,y-gridPhysicalSize.y*0.75,z-gridPhysicalSize.z*0.5)) < gridPhysicalSize.x*0.2) {
							//particlesVec.emplace_back(make_float3(x, y, z));
						}
					}
				}
			}
			particleCount = particlesVec.size();
			HANDLE_ERROR(cudaMalloc(&particles, particleCount * sizeof(Particle)));

			HANDLE_ERROR(cudaMemcpy(particles, particlesVec.data(), particleCount * sizeof(Particle), cudaMemcpyHostToDevice));

			numThreads = min(1024, particleCount);
			numBlocks = divUp(particleCount, numThreads);

			gridSize.x = ceil(gridPhysicalSize.x / kernelRadius);
			gridSize.y = ceil(gridPhysicalSize.y / kernelRadius);
			gridSize.z = ceil(gridPhysicalSize.z / kernelRadius);

			cellCount = gridSize.x * gridSize.y * gridSize.z;

			HANDLE_ERROR(cudaMalloc(&particleHashes, particleCount * sizeof(*particleHashes)));
			HANDLE_ERROR(cudaMalloc(&cellBegin, cellCount * sizeof(*cellBegin)));
			HANDLE_ERROR(cudaMalloc(&cellEnd, cellCount * sizeof(*cellEnd)));



			pointSprites = new PointSprites(particleCount);

			std::cout << "particle count : " << particleCount << std::endl;

			performSpatialHashing();
			calculateDensity(true);
			HANDLE_ERROR(cudaMemcpy(particlesVec.data(), particles,particleCount * sizeof(Particle), cudaMemcpyDeviceToHost));

			float totalRestDensity = 0;
			float maxDensity = 0;
			float minDensity = 99999;
			for (Particle& p : particlesVec) {
				totalRestDensity += p.density;
				maxDensity = max(maxDensity, p.density);
				minDensity = min(minDensity, p.density);
			}
			restDensity = totalRestDensity/(float)particleCount;


			float variance = 0;
			for (Particle& p : particlesVec) {
				variance += pow(p.density-restDensity,2);
			}
			variance /= (float)particleCount;


			std::cout << "spacing : " << particleSpacing << std::endl;
			std::cout << "kernel radius : " << kernelRadius << std::endl;
			std::cout << "rho0 : " << restDensity << std::endl;

			std::cout << "variance : " << variance << std::endl;


		}

		void updateVBO() {
			updatePositionsVBO << <numBlocks, numThreads >> > (particles, pointSprites->positionsDevice, particleCount);
		}

		

		void draw(glm::mat4& view, glm::mat4& projection, glm::vec3 cameraPos, float windowWidth, float windowHeight) {
			updateVBO();
			pointSprites->draw(view, projection, cameraPos, windowWidth, windowHeight,particleSpacing/2.0);
			container.draw(view, projection, cameraPos);
			
		}


		void calculateDensity(bool setAsRest = false) {
			calculateDensityImpl<< <numBlocks, numThreads >> >
				(particles, kernelRadius, particleCount, cellBegin, cellEnd, gridSize,kernelRadius,setAsRest);
		}


		void calculatePressure() {
			calculatePressureImpl << <numBlocks, numThreads >> > (particles, particleCount, restDensity, stiffness);
		}

		void addPressureForce() {
			calculateDensity();
			calculatePressure();
			addPressureForceImpl << <numBlocks, numThreads >> > 
				(particles, kernelRadius, particleCount, cellBegin, cellEnd, gridSize, kernelRadius);
		}

		void addExternalForces() {
			addExternalForcesImpl << <numBlocks, numThreads >> >(particles, particleCount);
		}

		void updatePosition() {
			updatePositionImpl << <numBlocks, numThreads >> >
				(particles, particleCount, gridPhysicalSize, particleSpacing,timestep / substeps);
		}

		void updateVelocity() {
			updateVelocityImpl << <numBlocks, numThreads >> >
				(particles, particleCount, timestep / substeps);
		}

		void simulationStep() {
			for (int i = 0; i < substeps; ++i) {
				//leap frog
				updatePosition();
				performSpatialHashing();


				addExternalForces();
				addPressureForce();
				updateVelocity();
			}
		}

		void performSpatialHashing() {
			HANDLE_ERROR(cudaMemset(cellBegin, 255, sizeof(*cellBegin) * cellCount));
			HANDLE_ERROR(cudaMemset(cellEnd, 255, sizeof(*cellEnd) * cellCount));

			calcHashImpl << <numBlocks, numThreads >> > (particles, particleHashes, kernelRadius, particleCount, gridSize);
			CHECK_CUDA_ERROR("calc hash");

			thrust::sort_by_key(thrust::device, particleHashes, particleHashes + particleCount, particles);

			findCellStartEndImpl << < numBlocks, numThreads >> > (particleHashes, cellBegin, cellEnd, particleCount);
			CHECK_CUDA_ERROR("find cell start end");

		}

	};
}

#endif