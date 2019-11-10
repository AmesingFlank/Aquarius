//
// Created by AmesingFlank on 2019-04-16.
//

#pragma once

#ifndef AQUARIUS_FLUID_2D_FLIP_CUH
#define AQUARIUS_FLUID_2D_FLIP_CUH

#include "../MAC_Grid_2D.cuh"
#include "../SPD_Solver.h"
#include <vector>
#include <utility>
#include "../GpuCommons.h"
#include "Fluid_2D.h"
#include <unordered_map>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include "FLuid_2D_common.cuh"
#include "FLuid_2D_kernels.cuh"




namespace Fluid_2D_FLIP {
	
    __device__ __host__
    struct Particle{
        float2 position = make_float2(0,0);
        float kind = 0;
        float2 velocity = make_float2(0,0);

        __device__ __host__
        Particle(){

        }
        Particle(float2 pos):position(pos){

        }
        Particle(float2 pos,float tag):position(pos),kind(tag){

        }
    };


    


    class Fluid : public Fluid_2D {
    public:
        const int sizeX = 40;
        const int sizeY = 20;
        const int cellCount = (sizeX+1)*(sizeY+1);


        const float cellPhysicalSize = 10.f / (float) sizeY;
        const float gravitationalAcceleration = 9.8;
        const float density = 1;
        MAC_Grid_2D grid = MAC_Grid_2D(sizeX, sizeY, cellPhysicalSize);

        Particle *particles;
        int particleCount;

        int numThreadsParticle, numBlocksParticle;
        int numThreadsCell, numBlocksCell;

        int particlesPerCell = 16;

        int* particleHashes;
        int * cellStart;
        int * cellEnd;

        Fluid(){
            init();
			initTextureImage(1024, 512);
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

            HANDLE_ERROR(cudaMalloc(&particleHashes, particleCount* sizeof(*particleHashes)));
            HANDLE_ERROR(cudaMalloc(&cellStart, cellCount* sizeof(*cellStart)));
            HANDLE_ERROR(cudaMalloc(&cellEnd, cellCount* sizeof(*cellEnd)));

            numThreadsParticle = min(1024, particleCount);
            numBlocksParticle = divUp(particleCount, numThreadsParticle);

            numThreadsCell = min(1024, sizeX * sizeY);
            numBlocksCell = divUp(sizeX * sizeY, numThreadsCell);

            std::cout << numThreadsCell << std::endl << numBlocksCell << std::endl;

			fixBoundary(grid);

        }

        

        virtual void simulationStep() override {
            float thisTimeStep = 0.01f;

            transferToGrid();
            grid.updateFluidCount();

            applyForces(thisTimeStep,grid,gravitationalAcceleration);
            fixBoundary(grid);

            //solvePressure(thisTimeStep);

			solvePressureJacobi(thisTimeStep,grid,100);


			updateVelocityWithPressure(thisTimeStep,grid);

            extrapolateVelocity(thisTimeStep,grid);

            transferToParticles();

            moveParticles(thisTimeStep);
        }


        void transferToGrid(){

			performSpatialHashing(particleHashes, particles, particleCount, cellPhysicalSize, sizeX, sizeY, numBlocksParticle, numThreadsParticle,cellStart,cellEnd,cellCount);

            resetAllCells << < numBlocksCell, numThreadsCell >> > (grid.cells, sizeX, sizeY, CONTENT_AIR);
            cudaDeviceSynchronize();
            CHECK_CUDA_ERROR("reset all cells");

            transferToCell<< < numBlocksCell, numThreadsCell >> >(grid.cells,sizeX,sizeY,cellPhysicalSize,cellStart,cellEnd,particles,particleCount);

        }

        void transferToParticles(){
            transferToParticlesImpl<<<numBlocksParticle,numThreadsParticle>>>(grid.cells,particles,particleCount,sizeX,sizeY,cellPhysicalSize);
            cudaDeviceSynchronize();
            CHECK_CUDA_ERROR("transfer to particles");
        }


        void moveParticles(float timeStep) {

            moveParticlesImpl << < numBlocksParticle, numThreadsParticle >> >
                                                      (timeStep, grid.cells, particles, particleCount, sizeX, sizeY, cellPhysicalSize);
            cudaDeviceSynchronize();
            CHECK_CUDA_ERROR("move particles");
            return;

        }

        


        virtual void draw(const DrawCommand& drawCommand) override {
			drawParticles(imageGPU, grid.sizeX, grid.sizeY, grid.cellPhysicalSize, particles, particleCount, numBlocksParticle,numThreadsParticle,imageSizeX, imageSizeY);
			//drawGrid(grid, imageGPU);
			drawImage();
			printGLError();

        }

        void createParticles(std::vector <Particle> &particlesHost, float2 centerPos, float tag = 0) {
            for (int particle = 0; particle < particlesPerCell; ++particle) {
                float xBias = (random0to1() - 0.5f) * cellPhysicalSize;
                float yBias = (random0to1() - 0.5f) * cellPhysicalSize;
                float2 particlePos = centerPos + make_float2(xBias, yBias);
                //xBias = 0;yBias=0;
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
                    thisCell.newVelocity = make_float2(0,0);
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
                        thisCell.newVelocity = make_float2(0,0);
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

#endif //AQUARIUS_FLUID_2D_Full_CUH
