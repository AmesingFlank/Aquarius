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
	__device__ __host__ struct Particle {
		float3 position = make_float3(0, 0, 0);
		float kind = 0;
		float3 velocity = make_float3(0, 0, 0);

		__device__ __host__
			Particle() {

		}
		Particle(float3 pos) :position(pos) {

		}
		Particle(float3 pos, float tag) :position(pos), kind(tag) {

		}
	};


	__global__  void transferToCellAccumPhase(Cell3D* cells, int sizeX, int sizeY, int sizeZ, float cellPhysicalSize, int* cellStart, int* cellEnd,
		Particle* particles, int particleCount);

	__global__  void transferToCellDividePhase(Cell3D* cells, int sizeX, int sizeY, int sizeZ, float cellPhysicalSize, int* cellStart, int* cellEnd,
		Particle* particles, int particleCount);


	__global__  void calcDensityImpl(Cell3D* cells, int sizeX, int sizeY, int sizeZ, float cellPhysicalSize, int* cellStart, int* cellEnd,
		Particle* particles, int particleCount);


	__global__  void transferToParticlesImpl(Cell3D* cells, Particle* particles, int particleCount, int sizeX, int sizeY, int sizeZ, float cellPhysicalSize);

	__global__  void moveParticlesImpl(float timeStep, Cell3D* cells, Particle* particles, int particleCount, int sizeX, int sizeY, int sizeZ, float cellPhysicalSize);

	__global__  void resetAllCells(Cell3D* cells, int sizeX, int sizeY, int sizeZ, float content);

	

	class Fluid :Fluid_3D {
	public:

		const int sizeXYZ = 32;

		const int sizeX = sizeXYZ;
		const int sizeY = sizeXYZ;
		const int sizeZ = sizeXYZ;


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

		Fluid();

		void init();

		virtual void simulationStep() override;


		void calcDensity();

		

		void transferToGrid();
		void transferToParticles();
		void moveParticles(float timeStep);

		virtual void draw(const DrawCommand& drawCommand) override;

		void createParticles(std::vector <Particle>& particlesHost, float3 centerPos, float tag = 0);

		void createSquareFluid(std::vector <Particle>& particlesHost, Cell3D* cellsTemp, int startIndex = 0);

		void createSphereFluid(std::vector <Particle>& particlesHost, Cell3D* cellsTemp, int startIndex = 0);

	};
}
