#pragma once

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

#define PBF_MAX_PHASES 4

namespace Fluid_3D_PBF {

	__host__ __device__ struct Particle {
		float3 position;
		float3 lastPosition;
		float3 deltaPosition;

		float3 velosity = make_float3(0, 0, 0);

		float lambda;

		float density;
		float restDensity;

		float s_corr;

		float volumeFractions[PBF_MAX_PHASES] = { 1 };

		__host__ __device__ Particle(float3 position_,int phase) :position(position_), lastPosition(position_) {
			for (int i = 0; i < PBF_MAX_PHASES;++i) {
				volumeFractions[i] = 0;
			}
			volumeFractions[phase] = 1;
		}

		__host__ __device__ Particle() {

		}
	};



	class Fluid : public Fluid_3D {
	public:
		int particleCount;
		int cellCount;

		Particle* particles;
		Particle* particlesCopy;
		int* particleIndices;
		int* particleHashes;
		int* cellBegin;
		int* cellEnd;

		float timestep = 0.033;
		float substeps = 4;

		float3 gridPhysicalSize = make_float3(10.f, 10.f, 10.f);

		int3 gridSize;

		float restDensity;

		float particleCountWhenFull = 3e5;

		float kernelRadiusToSpacingRatio = 2.01;

		float kernelRadius;
		float kernelRadius2;
		float kernelRadius3;
		float kernelRadius5;
		float kernelRadius6;
		float kernelRadius9;

		float particleSpacing;


		float solverIterations = 4;



		
		int numThreads, numBlocks;

		Container container = Container(glm::vec3(gridPhysicalSize.x, gridPhysicalSize.y, gridPhysicalSize.z));

		Skybox skybox = Skybox("resources/Skyboxes/GamlaStan2/", ".jpg");

		std::shared_ptr<Mesher> mesher;
		std::shared_ptr<FluidMeshRenderer> meshRenderer;
		std::shared_ptr<PointSprites> pointSprites;

		std::shared_ptr<FluidConfig3D> fluidConfig;

		int phaseCount;

		Fluid();


		virtual void draw(const DrawCommand& drawCommand) override;

		virtual void init(std::shared_ptr<FluidConfig> config) override;

		virtual void simulationStep() override;


		void computeRestDensity();
		void applyForces();
		void predictPosition();

		void computeDensityAndLambda(bool setDensityAsRestDensity);
		void computeDeltaPosition();
		void applyDeltaPosition();


		void updateVelocityAndPosition();




		void createSquareFluid(std::vector<Particle>& particlesVec, float3 minPos, float3 maxPos, int phase, int phaseCount);
		void createSphereFluid(std::vector<Particle>& particlesVec, float3 center, float radius, int phase, int phaseCount);



	};
}
