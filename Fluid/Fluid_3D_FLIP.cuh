#pragma once


#include "MAC_Grid_3D.cuh"
//#include "SPD_Solver.h"
#include <vector>
#include <utility>
#include "../Common/GpuCommons.h"
#include "Fluid_3D.cuh"
#include <unordered_map>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include "Fluid_3D_common.cuh"
#include "Fluid_3D_kernels.cuh"
#include "../Rendering/Renderer3D/PointSprites.h"
#include "../Rendering/Renderer3D/Skybox.h"
#include "../Rendering/Renderer3D/Mesher.cuh"
#include "../Rendering/Renderer3D/FluidMeshRenderer.cuh"
#include "../Rendering/Renderer3D/Container.h"


namespace Fluid_3D_FLIP{
	__device__ __host__ struct Particle {
		float3 position = make_float3(0, 0, 0);
		float3 velocity = make_float3(0, 0, 0);
		float4 volumeFractions = make_float4(0,0,0,0);

		__device__ __host__
		Particle() {
		}
		Particle(float3 pos,int phase = 0) :position(pos) {
			
			switch (phase) {
			case 0:
				volumeFractions.x = 1;
				break;
			case 1:
				volumeFractions.y = 1;
				break;
			case 2:
				volumeFractions.z = 1;
				break;
			case 3:
				volumeFractions.w = 1;
				break;
			}
			
		}
	};

	

	class Fluid :public Fluid_3D {
	public:

		float timestep;

		int sizeX;
		int sizeY;
		int sizeZ;

		float gridPhysicalSize = 10.f;

		


		int cellCount;


		float cellPhysicalSize;


		const float density = 1;

		Particle* particles;
		Particle* particlesCopy; //used for fast spatial hashing only
		int particleCount;

		int numThreadsParticle, numBlocksParticle;

		const int particlesPerCell = 8;

		int* particleHashes;
		int* particleIndices;

		int* cellStart;
		int* cellEnd;


		

		std::shared_ptr<PointSprites> pointSprites;

		

		std::shared_ptr<MAC_Grid_3D> grid;

		std::shared_ptr<Mesher> mesher;
		std::shared_ptr<FluidMeshRenderer> meshRenderer;




		float inkParticlesSpacing;
		Particle* inkParticles;
		int inkParticleCount;
		std::shared_ptr<PointSprites> pointSpritesInk;
		int numThreadsInkParticle, numBlocksInkParticle;


		FluidConfig config;

		Fluid();
		virtual ~Fluid() override;


		virtual void simulationStep() override;


		void countParticlesInCells();

		

		void transferToGrid();
		void transferToParticles();
		void moveParticles(float timeStep);

		virtual void draw(const DrawCommand& drawCommand) override;

		virtual void init(FluidConfig config) override;

		virtual glm::vec3 getCenter() override;
		virtual float getContainerSize() override;



		int createParticlesAt(std::vector <Particle>& particlesHost, float3 centerPos,std::function<bool(float3)> filter,float particleSpacing,int phase);

		void createSquareFluid(std::vector <Particle>& particlesHost, float3 minPos,float3 maxPos, int phase);

		void createSphereFluid(std::vector <Particle>& particlesHost, float3 center,float radius,int phase);





		void createSquareInk(std::vector <Particle>& particlesHost, float3 minPos, float3 maxPos,float spacing, int phase);

		void createSphereInk(std::vector <Particle>& particlesHost, float3 center, float radius, float spacing, int phase);

		void initInkRenderer();

		void transferToInkParticles();
		void moveInkParticles(float timeStep);

	};
}
