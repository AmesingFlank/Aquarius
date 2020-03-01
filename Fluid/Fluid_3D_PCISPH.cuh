#pragma once

#ifndef FLUID_3D_PCISPH
#define FLUID_3D_PCISPH

#include "../Common/GpuCommons.h"
#include <vector>
#include <algorithm>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include "WeightKernels.cuh"
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

		float timestep ;
		float substeps ;

		float3 gridPhysicalSize = make_float3(10.f, 10.f, 10.f);

		int3 gridSize;

		float restDensity;

		float particleCountWhenFull ;

		float kernelRadiusToSpacingRatio = 2;

		float stiffness;

		float kernelRadius;
		float kernelRadius2;
		float kernelRadius6;
		float kernelRadius9;

		float particleSpacing;

		int iterations;



		int numThreads, numBlocks;

		std::shared_ptr<Container> container;

		Skybox skybox = Skybox("resources/Skyboxes/GamlaStan2/",".jpg");

		std::shared_ptr<Mesher> mesher;
		std::shared_ptr<FluidMeshRenderer> meshRenderer;
		std::shared_ptr<PointSprites> pointSprites;

		FluidConfig fluidConfig;

		Fluid();
		virtual ~Fluid() override;

		virtual void draw(const DrawCommand& drawCommand) override;

		virtual void init(FluidConfig config);

		virtual glm::vec3 getCenter() override;

		void computeRestDensity();

		virtual void simulationStep() override;

		void computeExternalForces();

		void initPressure();

		void predictVelocityAndPosition();
		void predictDensityAndPressure();

		void computePressureForce();

		void computeNewVelocityAndPosition();


		void createSquareFluid(std::vector<Particle>& particlesVec, float3 minPos, float3 maxPos);
		void createSphereFluid(std::vector<Particle>& particlesVec, float3 center, float radius);



	};
}

#endif