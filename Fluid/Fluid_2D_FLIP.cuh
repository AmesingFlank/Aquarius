//
// Created by AmesingFlank on 2019-04-16.
//

#pragma once

#ifndef AQUARIUS_FLUID_2D_FLIP_CUH
#define AQUARIUS_FLUID_2D_FLIP_CUH

#include "MAC_Grid_2D.cuh"
#include "SPD_Solver.h"
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


	// used by FLIP
	template<typename Particle>
	__global__ inline void transferToCell(Cell2D* cells, int sizeX, int sizeY, float cellPhysicalSize, int* cellStart, int* cellEnd,
		Particle* particles, int particleCount) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= sizeX * sizeY) return;

		int y = index / sizeX;
		int x = index - y * sizeX;

		int cellID = x * (sizeY)+y;
		Cell2D& thisCell = get2D(cells, x, y);

		int cellsToCheck[9];
		for (int r = 0; r < 3; ++r) {
			for (int c = 0; c < 3; ++c) {
				cellsToCheck[c * 3 + r] = cellID + (r - 1) + (c - 1) * (sizeY);
			}
		}

		int cellCount = (sizeX) * (sizeY);

		float2 xVelocityPos = make_float2(x * cellPhysicalSize, (y + 0.5) * cellPhysicalSize);
		float2 yVelocityPos = make_float2((x + 0.5) * cellPhysicalSize, y * cellPhysicalSize);

		float totalWeightX = 0;
		float totalWeightY = 0;


		for (int cell : cellsToCheck) {
			if (cell >= 0 && cell < cellCount) {
				for (int j = cellStart[cell]; j <= cellEnd[cell]; ++j) {
					if (j >= 0 && j < particleCount) {
						const Particle& p = particles[j];
						float thisWeightX = trilinearHatKernel(p.position - xVelocityPos, cellPhysicalSize);
						float thisWeightY = trilinearHatKernel(p.position - yVelocityPos, cellPhysicalSize);

						thisCell.velocity.x += thisWeightX * p.velocity.x;
						thisCell.velocity.y += thisWeightY * p.velocity.y;

						totalWeightX += thisWeightX;
						totalWeightY += thisWeightY;
					}
				}
			}
		}

		if (totalWeightX > 0)
			thisCell.velocity.x /= totalWeightX;
		if (totalWeightY > 0)
			thisCell.velocity.y /= totalWeightY;
		thisCell.newVelocity = thisCell.velocity;

		for (int j = cellStart[cellID]; j <= cellEnd[cellID]; ++j) {
			if (j >= 0 && j < particleCount) {
				thisCell.content = CONTENT_FLUID;

				const Particle& p = particles[j];


				if (p.kind > 0) {
					thisCell.fluid1Count++;
				}
				else {
					thisCell.fluid0Count++;
				}
			}
		}
	}



	template<typename Particle>
	__global__ inline void transferToParticlesImpl(Cell2D* cells, Particle* particles, int particleCount, int sizeX, int sizeY, float cellPhysicalSize) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= particleCount) return;


		Particle& particle = particles[index];
		float2 newGridVelocity = MAC_Grid_2D::getPointNewVelocity(particle.position, cellPhysicalSize, sizeX, sizeY, cells);
		float2 oldGridVelocity = MAC_Grid_2D::getPointVelocity(particle.position, cellPhysicalSize, sizeX, sizeY, cells);
		float2 velocityChange = newGridVelocity - oldGridVelocity;
		particle.velocity += velocityChange; //FLIP

		//particle.velocity = newGridVelocity; //PIC

	}



	template<typename Particle>
	__global__ inline void moveParticlesImpl(float timeStep, Cell2D* cells, Particle* particles, int particleCount, int sizeX, int sizeY, float cellPhysicalSize) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= particleCount) return;

		Particle& particle = particles[index];
		float2 beginPos = particle.position;


		float2 u1 = MAC_Grid_2D::getPointNewVelocity(beginPos, cellPhysicalSize, sizeX, sizeY, cells);
		float2 u2 = MAC_Grid_2D::getPointNewVelocity(beginPos + timeStep * u1 / 2, cellPhysicalSize, sizeX, sizeY, cells);
		float2 u3 = MAC_Grid_2D::getPointNewVelocity(beginPos + timeStep * u2 * 3 / 4, cellPhysicalSize, sizeX, sizeY, cells);

		float2 destPos = beginPos + timeStep * (u1 * 2 / 9 + u2 * 3 / 9 + u3 * 4 / 9);

		//destPos = beginPos+particle.velocity*timeStep;


		destPos.x = max(0.0 + 1e-6, min(sizeX * cellPhysicalSize - 1e-6, destPos.x));
		destPos.y = max(0.0 + 1e-6, min(sizeY * cellPhysicalSize - 1e-6, destPos.y));

		particle.position = destPos;

	}

	__global__ inline void resetAllCells(Cell2D* cells, int sizeX, int sizeY, float content) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= sizeX * sizeY) return;

		int y = index / sizeX;
		int x = index - y * sizeX;

		Cell2D& thisCell = get2D(cells, x, y);

		thisCell.content = content;
		thisCell.velocity = make_float2(0, 0);
		thisCell.newVelocity = make_float2(0, 0);
		thisCell.fluid0Count = 0;
		thisCell.fluid1Count = 0;

	}
	
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
			drawParticles(imageGPU, sizeX*cellPhysicalSize,sizeY*cellPhysicalSize, particles, particleCount, numBlocksParticle,numThreadsParticle,imageSizeX, imageSizeY);
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
