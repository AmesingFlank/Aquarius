#pragma once

#ifndef FLUID_3D_PCISPH
#define FLUID_3D_PCISPH

#include "../GpuCommons.h"
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
#include "../Rendering/Renderer3D/Mesher.cuh"
#include "../Rendering/Renderer3D/FluidMeshRenderer.cuh"


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
	


	__global__ void computeExternalForcesImpl(Particle* particles, int particleCount);

	__global__ void initPressureImpl(Particle* particles, int particleCount);


	__global__ void predictVelocityAndPositionImpl(Particle* particles, int particleCount, float timestep, bool setAsActual, float spacing, float3 gridDimension);

	__global__ void predictDensityAndPressureImpl(Particle* particles, float cellSize, int particleCount, int* cellBegin, int* cellEnd, int3 gridSize, float kernelRadius, bool setAsRest, float timestep);

	__global__ void computePressureForceImpl(Particle* particles, float cellSize, int particleCount, int* cellBegin, int* cellEnd, int3 gridSize, float kernelRadius);

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

		float kernelRadiusToSpacingRatio = 2;

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


		Fluid();


		virtual void draw(const DrawCommand& drawCommand) override;

		virtual void init(std::shared_ptr<FluidConfig> config);

		void computeRestDensity();

		virtual void simulationStep() override;

		void computeExternalForces();

		void initPressure();
		bool hasBigError();
		void predictVelocityAndPosition();
		void predictDensityAndPressure();

		void computePressureForce();

		void computeNewVelocityAndPosition();


		void createSquareFluid(std::vector<Particle>& particlesVec, float3 minPos, float3 maxPos);
		void createSphereFluid(std::vector<Particle>& particlesVec, float3 center, float radius);


		//simulate as particles (same as the one in cuda samples). For debugging only.
		void simulateAsParticles();


	};
}

#endif