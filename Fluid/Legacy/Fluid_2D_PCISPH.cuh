
#ifndef AQUARIUS_FLUID_2D_PCISPH_CUH
#define AQUARIUS_FLUID_2D_PCISPH_CUH

#include "../Common/GpuCommons.h"
#include <vector>
#include <algorithm>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include "WeightKernels.cuh"
#include "Fluid_2D.h"

#include "FLuid_2D_common.cuh"
#include "FLuid_2D_kernels.cuh"

#define PRINT_INDEX 250

namespace Fluid_2D_PCISPH {

    __host__ __device__
    struct Particle{
        float2 position = make_float2(0,0);
        float2 velocity = make_float2(0,0);
        float2 newPosition = make_float2(0,0);
        float2 newVelocity = make_float2(0,0);

        float2 acceleration = make_float2(0,0);
        float pressure = 0;
        float2 pressureForce = make_float2(0,0);
        float2 otherForces = make_float2(0,0);
        float density = 0;

        __host__ __device__
        Particle(){

        }

        __host__ __device__
        Particle(float2 position_):position(position_){

        }

        int hash = 0;

        float lambda = 0;

		float kind = 0;
    };






    __global__
    inline
    void calcOtherForces(Particle *particles, int particleCount) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= particleCount) return;

        Particle &particle = particles[index];
        particle.otherForces = make_float2(0, -9.8);

        particle.pressure = 0;
        particle.pressureForce = make_float2(0, 0);

    }


    __global__
    inline
    void calcPositionVelocity(Particle *particles, int particleCount,
                              float gridBoundaryX, float gridBoundaryY, float timeStep, float cellPhysicalSize) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= particleCount) return;

        Particle &particle = particles[index];
        float2 newAcceleration = particle.pressureForce + particle.otherForces;
        float2 meanAcceleration = (newAcceleration + particle.acceleration) / 2.f;
        particle.acceleration = newAcceleration;

        float2 newVelocity = particle.velocity + timeStep * newAcceleration;
        particle.newVelocity = newVelocity;

        float2 meanVelocity = (newVelocity + particle.velocity) / 2.f;
        particle.newPosition = particle.position + newVelocity * timeStep;

        float damp = 0.00;

        if (particle.newPosition.x < 0 || particle.newPosition.x > gridBoundaryX) {
            particle.newVelocity.x *= -damp;
        }

        if (particle.newPosition.y < 0 || particle.newPosition.y > gridBoundaryY) {
            particle.newVelocity.y *= -damp;
        }

        particle.newPosition.x = min(gridBoundaryX, max(0.f, particle.newPosition.x));
        particle.newPosition.y = min(gridBoundaryY, max(0.f, particle.newPosition.y));


    }

    __global__
    inline
    void commitPositionVelocity(Particle *particles, int particleCount, float gridBoundaryX, float gridBoundaryY) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= particleCount) return;

        Particle &particle = particles[index];

        particle.position = particle.newPosition;
        particle.velocity = particle.newVelocity;


    }

    __global__
    inline
    void calcDensity(Particle *particles, int particleCount, int *cellStart, int *cellEnd,
                     int gridSizeX, int gridSizeY, float cellPhysicalSize, float restDensity, float SCP) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= particleCount) return;
        Particle &particle = particles[index];
        int cellX = particle.newPosition.x / cellPhysicalSize;
        int cellY = particle.newPosition.y / cellPhysicalSize;
        int cellID = cellY + cellX * gridSizeY;

        const float h = cellPhysicalSize;
        float density = restDensity * SCP;

        int start = cellStart[cellID];
        int end = cellEnd[cellID];

        //printf("%d\n",cellID);
        if (start > 0 && end > 0) {
            // printf("%d\n",end-start);
        }

        int cellsToCheck[9];
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                cellsToCheck[c * 3 + r] = cellID + (r - 1) + (c - 1) * gridSizeY;
            }
        }

        int cellCount = gridSizeX * gridSizeY;

        int nb = 0;

        for (int thisCell : cellsToCheck) {
            if (thisCell >= 0 && thisCell < cellCount) {
                for (int j = cellStart[thisCell]; j <= cellEnd[thisCell]; ++j) {
                    if (j >= 0 && j < particleCount && j != index &&
                        length(particle.newPosition - particles[j].newPosition) <= h) {
                        density += poly6(particle.newPosition - particles[j].newPosition, h);
                        nb += 1;
                    }
                }
            }
        }  


        particle.density = density;


    }

    __global__
    inline
    void calcPressure(Particle *particles, int particleCount, int *cellStart, int *cellEnd,
                      int gridSizeX, int gridSizeY, float cellPhysicalSize,
                      float restDensity, float timeStep) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= particleCount) return;
        Particle &particle = particles[index];
        int cellX = particle.newPosition.x / cellPhysicalSize;
        int cellY = particle.newPosition.y / cellPhysicalSize;
        int cellID = cellY + cellX * gridSizeY;

        const float h = cellPhysicalSize;

        float deltaDensity = particle.density - restDensity;

        float coeff = 0;

        coeff = 1;
        particle.pressure += coeff * deltaDensity;
		if (index == PRINT_INDEX) {
			//printf("rho: %f\n", particle.density);
		}
        //

    }


    __global__
    inline
    void calcPressureForce(Particle *particles, int particleCount, int *cellStart, int *cellEnd,
                           int gridSizeX, int gridSizeY, float cellPhysicalSize, float restDensity) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= particleCount) return;
        Particle &particle = particles[index];
        int cellX = particle.newPosition.x / cellPhysicalSize;
        int cellY = particle.newPosition.y / cellPhysicalSize;
        int cellID = cellY + cellX * gridSizeY;

        const float h = cellPhysicalSize;
        float2 Fp = make_float2(0, 0);

        int cellsToCheck[9];
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                cellsToCheck[c * 3 + r] = cellID + (r - 1) + (c - 1) * gridSizeY;
            }
        }

        int cellCount = gridSizeX * gridSizeY;

        for (int thisCell : cellsToCheck) {
            if (thisCell >= 0 && thisCell < cellCount) {
                for (int j = cellStart[thisCell]; j <= cellEnd[thisCell]; ++j) {
                    if (j >= 0 && j < particleCount && j != index) {
                        if (particle.newPosition.x == particles[j].newPosition.x) {
                            if (particle.newPosition.y == particles[j].newPosition.y) {
                                particle.newPosition.x += 0.001;
                                particle.newPosition.y += 0.001;
                            }
                        }
                        float2 weight = spikey_grad(particle.newPosition - particles[j].newPosition, h);
                        if (weight.x != weight.x || weight.y != weight.y) {
                            printf("shit, weight is nan\n");
                        }
                        Fp -= weight * (particle.pressure / pow(particle.density, 2) +
                                        particles[j].pressure / pow(particles[j].density, 2));
                        //printf("%f\n",weight.x);
                    }
                }
            }
        }

        particle.pressureForce = Fp;

        if (particle.newPosition.x == 0 && particle.pressureForce.x <= 0) {
            particle.pressureForce.x = 0.00001;
        }
        if (particle.newPosition.y == 0 && particle.pressureForce.y <= 0) {
            particle.pressureForce.y = 0.00001;
        }

    }



    class Fluid:public Fluid_2D{
    public:

        const float gridBoundaryX = 6.f;
        const float gridBoundaryY = 3.f;


        const float restDensity = 100;
        const float particleRadius = sqrt(1/restDensity/M_PI);
        const float SCP = 0.7;
        const float kernalRadius = sqrt(4.0/(M_PI*restDensity*SCP));

		float particleSeperation;

		void calculateParticleSeperation() {
			int neighboursPerSide = 1;
			int sides = 4;
			double l = 0;
			double r = kernalRadius;
			while ((r - l) / kernalRadius > 1e-6) {
				double m = (r + l) / 2;
				float contributionPerSide = 0;
				for (int i = 1; i <= neighboursPerSide; ++i) {
					contributionPerSide += poly6(make_float2(m*i, 0), kernalRadius);
				}
				float density = poly6(make_float2(0, 0), kernalRadius) + contributionPerSide*sides;
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
			particleSeperation = (r + l) / 2;
			std::cout << "found sep! : " << particleSeperation<< std::endl;

		}

        const float cellPhysicalSize = kernalRadius;

        const int gridSizeX = ceil(gridBoundaryX/cellPhysicalSize);
        const int gridSizeY = ceil(gridBoundaryY/cellPhysicalSize);
        const int cellCount = gridSizeX*gridSizeY;


        int * cellStart;
        int * cellEnd;

        Particle* particles;
        int* particleHashes;
        int particleCount = 0;


        int numThreads, numBlocks;


        Fluid(){
            std::cout<<"kernal radius "<<kernalRadius<<std::endl;
            std::cout<<"particle radius "<<particleRadius<<std::endl;

            std::cout<<"gridSizeX: "<<gridSizeX<<"     gridSizeY:"<<gridSizeY<<std::endl;
            std::cout<<"self contributed density: "<<poly6(make_float2(0,0),kernalRadius)<<std::endl;

            initFluid();
			initTextureImage(512,256);
        }


        void createParticles(std::vector<Particle>& particleList, float2 centerPos){
            for (int particle = 0; particle < 1 ; ++particle) {
                float xBias = (random0to1()-0.5f)*particleRadius;
                float yBias = (random0to1()-0.5f)*particleRadius;
                xBias = 0;yBias = 0;
                float2 particlePos = centerPos+make_float2(xBias,yBias);
                particleList.emplace_back(particlePos);
            }
        }

        void initFluid(){
            std::vector<Particle> allParticles;

			calculateParticleSeperation();

            for(float x = 0;x<gridBoundaryX/particleSeperation;x+=1){
                for (float y = 0; y < gridBoundaryY/ particleSeperation; y+=1) {
                    float2 pos = make_float2( (x+0.5f)* particleSeperation, (y+0.5f)* particleSeperation);
                    if(pos.y < gridBoundaryY* 0.33){
                        createParticles(allParticles,pos);
                    }
                    else if(pow(pos.x- 0.5*gridBoundaryX,2)+pow(pos.y- 0.7*gridBoundaryY ,2) <= pow(0.2*gridBoundaryY,2) ){
                        createParticles(allParticles,pos);
                    }

                }
            }
            particleCount = allParticles.size();
            std::cout<<"particles:"<<particleCount<<std::endl;


            Particle* newParticles = new Particle[particleCount];
            for(int i = 0;i<particleCount;++i){
                newParticles[i] = allParticles[i];
            }
            HANDLE_ERROR(cudaMalloc(&particles, particleCount* sizeof(*particles)));
            HANDLE_ERROR(cudaMemcpy(particles, newParticles, particleCount* sizeof(Particle), cudaMemcpyHostToDevice));
            delete []newParticles;


            HANDLE_ERROR(cudaMalloc(&cellStart, cellCount* sizeof(*cellStart)));
            HANDLE_ERROR(cudaMalloc(&cellEnd, cellCount* sizeof(*cellEnd)));

            HANDLE_ERROR(cudaMalloc(&particleHashes, particleCount* sizeof(*particleHashes)));

            HANDLE_ERROR(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, numBlocks*numThreads*1024));

            numThreads = min(1024,particleCount);
            numBlocks = divUp(particleCount,numThreads);

            //performSpatialHashing();
            //calcDensity<<< numBlocks, numThreads >>>(particles,particleCount,cellStart,cellEnd,gridSizeX,gridSizeY,cellPhysicalSize,restDensity);
            //CHECK_CUDA_ERROR("calcDensity 0");
        }

        virtual void simulationStep() override{
            float thisTimeStep = 0.01f;

			performSpatialHashing(particleHashes, particles, particleCount, cellPhysicalSize, gridSizeX, gridSizeY, numBlocks, numThreads, cellStart, cellEnd, cellCount);
            calcOtherForces<<< numBlocks, numThreads >>>(particles,particleCount);
            CHECK_CUDA_ERROR("calcOtherForces");
            for (int i = 0 ; i < 20 ; ++i) {
                calcPositionVelocity<<< numBlocks, numThreads >>>(particles,particleCount,gridBoundaryX,gridBoundaryY,thisTimeStep,cellPhysicalSize);
                CHECK_CUDA_ERROR("calcPositionVelocity");

                //performSpatialHashing(1);

                calcDensity<<< numBlocks, numThreads >>>(particles,particleCount,cellStart,cellEnd,gridSizeX,gridSizeY,cellPhysicalSize,restDensity,SCP);
                CHECK_CUDA_ERROR("calcDensity");

                calcPressure<<< numBlocks, numThreads >>>(particles,particleCount,cellStart,cellEnd,gridSizeX,gridSizeY,cellPhysicalSize,restDensity,thisTimeStep);
                CHECK_CUDA_ERROR("calcPressure");

                calcPressureForce<<< numBlocks, numThreads >>>(particles,particleCount,cellStart,cellEnd,gridSizeX,gridSizeY,cellPhysicalSize,restDensity);
                CHECK_CUDA_ERROR("calcPressureForce");

            }
            calcPositionVelocity<<< numBlocks, numThreads >>>(particles,particleCount,gridBoundaryX,gridBoundaryY,thisTimeStep,cellPhysicalSize);
            commitPositionVelocity<<< numBlocks, numThreads >>>(particles,particleCount,gridBoundaryX,gridBoundaryY);
            CHECK_CUDA_ERROR("calcPositionVelocity");

            //std::cout<<"finished one step"<<std::endl<<std::endl;
        }


		virtual void draw(const DrawCommand& drawCommand) override {
			drawParticles(imageGPU, gridBoundaryX, gridBoundaryY, particles, particleCount, numBlocks, numThreads, imageSizeX, imageSizeY);

			drawImage();
			printGLError();

		}

    };

}


#endif //AQUARIUS_FLUID_2D_PCISPH_CUH
