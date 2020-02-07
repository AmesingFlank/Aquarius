#pragma once


#include "MAC_Grid_3D.cuh"
#include "SPD_Solver.h"
#include <vector>
#include <utility>
#include "../Common/GpuCommons.h"
#include "Fluid_3D.cuh"
#include <unordered_map>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include "FLuid_3D_common.cuh"
#include "FLuid_3D_kernels.cuh"
#include "../Rendering/Renderer3D/PointSprites.h"
#include "../Rendering/Renderer3D/Container.h"
#include "../Rendering/Renderer3D/Skybox.h"
#include "../Rendering/Renderer3D/Mesher.cuh"
#include "../Rendering/Renderer3D/FluidMeshRenderer.cuh"

namespace Fluid_3D_FLIP{
	__device__ __host__ struct Particle {
		float3 position = make_float3(0, 0, 0);
		float3 velocity = make_float3(0, 0, 0);

		__device__ __host__
			Particle() {

		}
		Particle(float3 pos) :position(pos) {

		}
	};

	

	class Fluid :public Fluid_3D {
	public:

		float timestep = 0.033f;

		int sizeX;
		int sizeY;
		int sizeZ;


		int cellCount;


		float cellPhysicalSize;

		const float gravitationalAcceleration = 9.8;
		const float density = 1;

		Particle* particles;
		Particle* particlesCopy; //used for fast spatial hashing only
		int particleCount;

		int numThreadsParticle, numBlocksParticle;
		int numThreadsCell, numBlocksCell;

		const int particlesPerCell = 8;

		int* particleHashes;
		int* particleIndices;

		int* cellStart;
		int* cellEnd;


		Skybox skybox = Skybox("resources/Skyboxes/GamlaStan2/", ".jpg");

		std::shared_ptr<PointSprites> pointSprites;

		

		std::shared_ptr<Container> container;
		std::shared_ptr<MAC_Grid_3D> grid;

		std::shared_ptr<Mesher> mesher;
		std::shared_ptr<FluidMeshRenderer> meshRenderer;


		std::shared_ptr<FluidConfig3D> fluidConfig;


		float inkParticlesSpacing;
		Particle* inkParticles;
		int inkParticleCount;
		std::shared_ptr<PointSprites> pointSpritesInk;
		int numThreadsInkParticle, numBlocksInkParticle;

		Fluid();



		virtual void simulationStep() override;


		void calcDensity();

		

		void transferToGrid();
		void transferToParticles();
		void moveParticles(float timeStep);

		virtual void draw(const DrawCommand& drawCommand) override;

		virtual void init(std::shared_ptr<FluidConfig> config) override;



		int createParticlesAt(std::vector <Particle>& particlesHost, float3 centerPos,std::function<bool(float3)> filter,float particleSpacing);

		void createSquareFluid(std::vector <Particle>& particlesHost, Cell3D* cellsTemp, float3 minPos,float3 maxPos, int startIndex);

		void createSphereFluid(std::vector <Particle>& particlesHost, Cell3D* cellsTemp,float3 center,float radius, int startIndex);





		void createSquareInk(std::vector <Particle>& particlesHost, float3 minPos, float3 maxPos,float spacing);

		void createSphereInk(std::vector <Particle>& particlesHost, float3 center, float radius, float spacing);

		void initInkRenderer();

		void drawInk(const DrawCommand& drawCommand);
		void transferToInkParticles();
		void moveInkParticles(float timeStep);

	};
}
