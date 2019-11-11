//
// Created by AmesingFlank on 2019-04-16.
//

#ifndef AQUARIUS_FLUID_2D_SEMILAGRANGE_CUH
#define AQUARIUS_FLUID_2D_SEMILAGRANGE_CUH

#include "MAC_Grid_2D.cuh"
#include "SPD_Solver.h"
#include <vector>
#include <utility>
#include "GpuCommons.h"
#include "Fluid_2D.h"
#include <unordered_map>

#include "FLuid_2D_common.cuh"
#include "FLuid_2D_kernels.cuh"


namespace Fluid_2D_SemiLagrange {

    
    __device__ __host__
    struct Particle{
        float2 position;
        float kind = 0;
        Particle(){

        }
        Particle(float2 pos):position(pos){

        }
        Particle(float2 pos,float tag):position(pos),kind(tag){

        }
    };


	__global__ inline  void setAllContent(Cell2D* cells, int sizeX, int sizeY, float content) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= (sizeX + 1) * (sizeY + 1)) return;

		int x = index / (sizeY + 1);
		int y = index - x * (sizeY + 1);
		get2D(cells, x, y).content_new = content;
	}


    __global__
    inline void advectVelocityImpl(Cell2D *cells, int sizeX, int sizeY, float timeStep, float gravitationalAcceleration,
                            float cellPhysicalSize) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= (sizeX+1) * (sizeY+1)) return;

        int x = index / (sizeY+1);
        int y = index - x * (sizeY+1);

        if (get2D(cells,x,y).content == CONTENT_AIR) return;
        float2 thisVelocity = MAC_Grid_2D::getCellVelocity(x, y, sizeX, sizeY, cells);
        float2 thisPos = MAC_Grid_2D::getPhysicalPos(x, y, cellPhysicalSize);

        float2 u1 = MAC_Grid_2D::getPointVelocity(thisPos, cellPhysicalSize, sizeX, sizeY, cells);
        float2 u2 = MAC_Grid_2D::getPointVelocity(thisPos-timeStep*u1/2, cellPhysicalSize, sizeX, sizeY, cells);
        float2 u3 = MAC_Grid_2D::getPointVelocity(thisPos-timeStep*u2*3/4, cellPhysicalSize, sizeX, sizeY, cells);

        float2 sourcePos = thisPos - timeStep* ( u1*2/9 + u2*3/9 + u3*4/9 );


        float2 sourceVelocity =
                MAC_Grid_2D::getPointVelocity(sourcePos, cellPhysicalSize, sizeX, sizeY, cells);
        get2D(cells,x,y).newVelocity = sourceVelocity;
        if (y + 1 <= sizeY && get2D(cells,x,y + 1).content == CONTENT_AIR) {
            get2D(cells,x,y + 1).newVelocity.y = sourceVelocity.y;
        }
        if (x + 1 <= sizeX && get2D(cells,x + 1,y).content == CONTENT_AIR) {
            get2D(cells,x + 1,y).newVelocity.x = sourceVelocity.x;
        }
    }

    __global__
    inline  void moveParticlesImpl(float timeStep, Cell2D *cells, Particle *particles, int particleCount, int sizeX, int sizeY,
                      float cellPhysicalSize) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= particleCount) return;

        Particle &particle = particles[index];
        float2 beginPos = particle.position;



        float2 u1 = MAC_Grid_2D::getPointVelocity(beginPos, cellPhysicalSize, sizeX, sizeY, cells);
        float2 u2 = MAC_Grid_2D::getPointVelocity(beginPos+timeStep*u1/2, cellPhysicalSize, sizeX, sizeY, cells);
        float2 u3 = MAC_Grid_2D::getPointVelocity(beginPos+timeStep*u2*3/4, cellPhysicalSize, sizeX, sizeY, cells);

        float2 destPos = beginPos + timeStep* ( u1*2/9 + u2*3/9 + u3*4/9 );


        int destCellX = floor(destPos.x / cellPhysicalSize);
        int destCellY = floor(destPos.y / cellPhysicalSize);
        destCellX = max(min(destCellX, sizeX - 1), 0);
        destCellY = max(min(destCellY, sizeY - 1), 0);
        get2D(cells,destCellX,destCellY).content_new = CONTENT_FLUID;
        particle.position = destPos;


        if (particle.kind > 0) {
            get2D(cells,destCellX,destCellY).fluid1Count += 1;
        } else
            get2D(cells,destCellX,destCellY).fluid0Count += 1;
    }



    __global__
    inline void commitVelocityChanges(Cell2D *cells, int sizeX, int sizeY) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= sizeX * sizeY) return;

        int y = index / sizeX;
        int x = index - y * sizeX;

        Cell2D &thisCell = get2D(cells,x,y);
        thisCell.velocity = thisCell.newVelocity;
    }

    class Fluid : public Fluid_2D {
    public:
        const int sizeX = 256;
        const int sizeY = 128;
        const int cellCount = (sizeX+1)*(sizeY+1);
        const float cellPhysicalSize = 10.f / (float) sizeY;
        const float gravitationalAcceleration = 9.8;
        const float density = 1;
        MAC_Grid_2D grid = MAC_Grid_2D(sizeX, sizeY, cellPhysicalSize);

        Particle *particles;
        int particleCount;

        int numThreadsParticle, numBlocksParticle;
        int numThreadsCell, numBlocksCell;

        Fluid() {
            init();
			initTextureImage(sizeX,sizeY);
        }

        virtual void simulationStep() override {
            float thisTimeStep = 0.05f;

            //extrapolateVelocity(thisTimeStep);

            advectVelocity(thisTimeStep);


            applyForces(thisTimeStep,grid,gravitationalAcceleration);
            fixBoundary(grid);

            solvePressure(thisTimeStep,grid);

			updateVelocityWithPressure(thisTimeStep, grid);

			extrapolateVelocity(thisTimeStep, grid);

			commitVelocityChanges << <numBlocksCell, numThreadsCell >> > (grid.cells, sizeX, sizeY);

            moveParticles(thisTimeStep);
            grid.commitContentChanges();

        }

		virtual void draw(const DrawCommand& drawCommand) override {
			//drawParticles(imageGPU, grid.sizeX, grid.sizeY, grid.cellPhysicalSize, particles, particleCount, numBlocksParticle, numThreadsParticle, imageSizeX, imageSizeY);
			drawGrid(grid, imageGPU);
			drawImage();
			printGLError();

		}

        void init() {


            //set everything to air first

            Cell2D *cellsTemp = grid.copyCellsToHost();


            grid.fluidCount = 0;
            std::vector <Particle> particlesHost;
            createSquareFluid(particlesHost, cellsTemp);
            createSphereFluid(particlesHost, cellsTemp, grid.fluidCount);
            particleCount = particlesHost.size();

            grid.copyCellsToDevice(cellsTemp);
            delete []cellsTemp;

            HANDLE_ERROR(cudaMalloc(&particles, particleCount * sizeof(Particle)));
            Particle *particlesHostToCopy = new Particle[particleCount];
            for (int i = 0; i < particleCount; ++i) {
                particlesHostToCopy[i] = particlesHost[i];
            }
            HANDLE_ERROR(cudaMemcpy(particles, particlesHostToCopy, particleCount * sizeof(Particle),
                                    cudaMemcpyHostToDevice));
            delete[] particlesHostToCopy;

            numThreadsParticle = min(1024, particleCount);
            numBlocksParticle = divUp(particleCount, numThreadsParticle);

            numThreadsCell = min(1024, cellCount);
            numBlocksCell = divUp(cellCount, numThreadsCell);

            std::cout << numThreadsCell << std::endl << numBlocksCell << std::endl;

            fixBoundary(grid);

        }

        void advectVelocity(float timeStep) {
            advectVelocityImpl << < numBlocksCell, numThreadsCell >> >
                                                   (grid.cells, sizeX, sizeY, timeStep, gravitationalAcceleration, cellPhysicalSize);
            cudaDeviceSynchronize();
            CHECK_CUDA_ERROR("advect velocity");

        }

        void moveParticles(float timeStep) {

			setAllContent << < numBlocksCell, numThreadsCell >> > (grid.cells, sizeX, sizeY, CONTENT_AIR);
			CHECK_CUDA_ERROR("set all to air");

            moveParticlesImpl << < numBlocksParticle, numThreadsParticle >> >
                                                      (timeStep, grid.cells, particles, particleCount, sizeX, sizeY, cellPhysicalSize);
            cudaDeviceSynchronize();
            CHECK_CUDA_ERROR("move particles");
            return;

        }



        void createParticles(std::vector <Particle> &particlesHost, float2 centerPos, float tag = 0) {
            for (int particle = 0; particle < 8; ++particle) {
                float xBias = (random0to1() - 0.5f) * cellPhysicalSize;
                float yBias = (random0to1() - 0.5f) * cellPhysicalSize;
                float2 particlePos = centerPos + make_float2(xBias, yBias);
                particlesHost.emplace_back(particlePos,tag);
            }
        }

        void createSquareFluid(std::vector <Particle> &particlesHost, Cell2D *cellsTemp, int startIndex = 0) {
            int index = startIndex;
            for (int y = 0 * sizeY; y < 0.2 * sizeY; ++y) {
                for (int x = 0 * sizeX; x < 1 * sizeX; ++x) {

                    Cell2D &thisCell = cellsTemp[x * (sizeY + 1) + y];

                    thisCell.velocity.x = 0;
                    thisCell.velocity.y = 0;
                    thisCell.content = CONTENT_FLUID;
                    thisCell.fluidIndex = index;
                    ++index;
                    float2 thisPos = MAC_Grid_2D::getPhysicalPos(x, y, cellPhysicalSize);
                    createParticles(particlesHost, thisPos, 0);
                }
            }

            grid.fluidCount = index;
        }

        void createSphereFluid(std::vector <Particle> &particlesHost, Cell2D *cellsTemp, int startIndex = 0) {
            int index = startIndex;
            for (int y = 0 * sizeY; y < 1 * sizeY; ++y) {
                for (int x = 0 * sizeX; x < 1 * sizeX; ++x) {
                    if (pow(x - 0.5 * sizeX, 2) + pow(y - 0.7 * sizeY, 2) <= pow(0.2 * sizeY, 2)) {

                        Cell2D &thisCell = cellsTemp[x * (sizeY + 1) + y];

                        thisCell.velocity.x = 0;
                        thisCell.velocity.y = 0;
                        thisCell.content = CONTENT_FLUID;
                        thisCell.fluidIndex = index;
                        ++index;

                        float2 thisPos = MAC_Grid_2D::getPhysicalPos(x, y, cellPhysicalSize);
                        createParticles(particlesHost, thisPos, 1);
                    }
                }
            }

            grid.fluidCount = index;
        }

    };
}

#endif //AQUARIUS_FLUID_2D_SEMILAGRANGE_CUH
