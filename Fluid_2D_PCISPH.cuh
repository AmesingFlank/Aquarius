//
// Created by AmesingFlank on 2019-07-01.
//

#ifndef AQUARIUS_FLUID_2D_PCISPH_CUH
#define AQUARIUS_FLUID_2D_PCISPH_CUH

#include "CudaCommons.h"
#include <vector>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include "WeightKernels.h"
#include "Fluid_2D.h"

#define PRINT_INDEX 250

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
};


namespace Fluid_2D_PCISPH_Impl {

    __global__
    void calcHashImpl(int *particleHashes,  // output
                      Particle *particles,               // input: positions
                      int particleCount,
                      float cellPhysicalSize, int gridSizeX, int gridSizeY, int version) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;

        if (index >= particleCount) return;

        Particle &p = particles[index];

        float2 pos = p.position;
        if (version != 0) {
            pos = p.newPosition;
        }
        // get address in grid
        int x = pos.x / cellPhysicalSize;
        int y = pos.y / cellPhysicalSize;
        int hash = x * gridSizeY + y;

        particleHashes[index] = hash;

        //printf("p %d --> c %d\n",index,hash);
    }

    __global__
    void findCellStartEndImpl(int *particleHashes,
                              int *cellStart, int *cellEnd,
                              int particleCount) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;

        if (index >= particleCount) return;

        int thisHash = particleHashes[index];


        if (index == 0 || particleHashes[index - 1] < thisHash) {
            cellStart[thisHash] = index;
        }

        if (index == particleCount - 1 || particleHashes[index + 1] > thisHash) {
            cellEnd[thisHash] = index;
        }
    }


    __global__
    void calcOtherForces(Particle *particles, int particleCount) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= particleCount) return;

        Particle &particle = particles[index];
        particle.otherForces = make_float2(0, -9.8);

        particle.pressure = 0;
        particle.pressureForce = make_float2(0, 0);

    }


    __global__
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
    void commitPositionVelocity(Particle *particles, int particleCount, float gridBoundaryX, float gridBoundaryY) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= particleCount) return;

        Particle &particle = particles[index];

        particle.position = particle.newPosition;
        particle.velocity = particle.newVelocity;


    }

    __global__
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

        if (index == PRINT_INDEX)
            printf("neibours : %d\n", nb);


        particle.density = density;
        if (density == 0) {
            printf("shit density is 0 %d\n");
        }

    }

    __global__
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

        if (index == PRINT_INDEX) printf("delta density %f\n", deltaDensity);

        float coeff = 0;

        coeff = 1;
        particle.pressure += coeff * deltaDensity;
        //printf("%f\n",coeff);

    }


    __global__
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

        if (index == PRINT_INDEX) printf("                                    Fp: x: %f ,    y: %f \n", Fp.x, Fp.y);

        particle.pressureForce = Fp;

        if (particle.newPosition.x == 0 && particle.pressureForce.x <= 0) {
            particle.pressureForce.x = 0.00001;
        }
        if (particle.newPosition.y == 0 && particle.pressureForce.y <= 0) {
            particle.pressureForce.y = 0.00001;
        }

    }


    __global__
    void updateTextureImpl(Particle *particles, int particleCount, float texCellPhysicalSize,
                           int texSizeX, unsigned char *result) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= particleCount) return;
        Particle &particle = particles[index];
        int cellX = particle.position.x / texCellPhysicalSize;
        int cellY = particle.position.y / texCellPhysicalSize;
        int cellID = cellY * texSizeX + cellX;

        unsigned char *base = result + cellID * 4;
        base[0] = 0;
        base[1] = 0;
        base[2] = 255;
        base[3] = 255;
    }

}


class Fluid_2D_PCISPH:public Fluid_2D{
public:



    const float gridBoundaryX = 6.f;
    const float gridBoundaryY = 3.f;


    const float restDensity = 100;
    const float partilceRadius = sqrt(1/restDensity/M_PI);
    const float SCP = 0.8;
    const float kernalRadius = sqrt(4.0/(M_PI*restDensity*SCP));


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


    Fluid_2D_PCISPH(){
        std::cout<<"kernal radius "<<kernalRadius<<std::endl;
        std::cout<<"particle radius "<<partilceRadius<<std::endl;

        std::cout<<"gridSizeX: "<<gridSizeX<<"     gridSizeY:"<<gridSizeY<<std::endl;
        std::cout<<"self contributed density: "<<poly6(make_float2(0,0),kernalRadius)<<std::endl;

        initFluid();
    }

    void performSpatialHashing(int version = 0){
        Fluid_2D_PCISPH_Impl::calcHashImpl<<< numBlocks, numThreads >>>(particleHashes,particles,particleCount,cellPhysicalSize,gridSizeX,gridSizeY,version);

        CHECK_CUDA_ERROR("calc hash");
        thrust::sort_by_key(thrust::device, particleHashes, particleHashes + particleCount, particles);

        /*
        for (int i = 0; i <particleCount ; ++i) {
            int thisHash = 0;
            HANDLE_ERROR(cudaMemcpy(&thisHash, particleHashes+i, sizeof(int), cudaMemcpyDeviceToHost));
            std::cout<<"hash of "<<i<<" is "<<thisHash<<std::endl;
        }
         */

        HANDLE_ERROR(cudaMemset(cellStart,-1,cellCount*sizeof(*cellStart)));
        HANDLE_ERROR(cudaMemset(cellEnd,-1,cellCount*sizeof(*cellEnd)));
        Fluid_2D_PCISPH_Impl::findCellStartEndImpl<<< numBlocks, numThreads >>>(particleHashes,cellStart,cellEnd,particleCount);
        CHECK_CUDA_ERROR("find cell start end");


        //std::cout<<"finished spatial hashing"<<std::endl;

    }

    void createParticles(std::vector<Particle>& particleList, float2 centerPos){
        for (int particle = 0; particle < 1 ; ++particle) {
            float xBias = (random0to1()-0.5f)*partilceRadius;
            float yBias = (random0to1()-0.5f)*partilceRadius;
            //xBias = 0;yBias = 0;
            float2 particlePos = centerPos+make_float2(xBias,yBias);
            particleList.emplace_back(particlePos);
        }
    }

    void initFluid(){
        std::vector<Particle> allParticles;

        for(int x = 0;x<gridBoundaryX/partilceRadius;x+=3){
            for (int y = 0; y < gridBoundaryY/partilceRadius ; y+=3) {
                float2 pos = make_float2( (x+0.5f)*partilceRadius, (y+0.5f)*partilceRadius );
                if(pos.y < gridBoundaryY* 0.93){
                    createParticles(allParticles,pos);
                }
                else if(pow(pos.x- 0.5*gridBoundaryX,2)+pow(pos.y- 0.7*gridBoundaryY ,2) <= pow(0.2*gridBoundaryY,2) ){
                    //createParticles(allParticles,pos);
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

    void simulationStep(float totalTime){
        float thisTimeStep = 0.01f;

        performSpatialHashing();
        Fluid_2D_PCISPH_Impl::calcOtherForces<<< numBlocks, numThreads >>>(particles,particleCount);
        CHECK_CUDA_ERROR("calcOtherForces");
        for (int i = 0 ; i < 20 ; ++i) {
            Fluid_2D_PCISPH_Impl::calcPositionVelocity<<< numBlocks, numThreads >>>(particles,particleCount,gridBoundaryX,gridBoundaryY,thisTimeStep,cellPhysicalSize);
            CHECK_CUDA_ERROR("calcPositionVelocity");

            //performSpatialHashing(1);

            Fluid_2D_PCISPH_Impl:: calcDensity<<< numBlocks, numThreads >>>(particles,particleCount,cellStart,cellEnd,gridSizeX,gridSizeY,cellPhysicalSize,restDensity,SCP);
            CHECK_CUDA_ERROR("calcDensity");

            Fluid_2D_PCISPH_Impl::calcPressure<<< numBlocks, numThreads >>>(particles,particleCount,cellStart,cellEnd,gridSizeX,gridSizeY,cellPhysicalSize,restDensity,thisTimeStep);
            CHECK_CUDA_ERROR("calcPressure");

            Fluid_2D_PCISPH_Impl::calcPressureForce<<< numBlocks, numThreads >>>(particles,particleCount,cellStart,cellEnd,gridSizeX,gridSizeY,cellPhysicalSize,restDensity);
            CHECK_CUDA_ERROR("calcPressureForce");

        }
        Fluid_2D_PCISPH_Impl::calcPositionVelocity<<< numBlocks, numThreads >>>(particles,particleCount,gridBoundaryX,gridBoundaryY,thisTimeStep,cellPhysicalSize);
        Fluid_2D_PCISPH_Impl::commitPositionVelocity<<< numBlocks, numThreads >>>(particles,particleCount,gridBoundaryX,gridBoundaryY);
        CHECK_CUDA_ERROR("calcPositionVelocity");
        updateTexture();
        std::cout<<"finished one step"<<std::endl;
    }


    virtual void updateTexture()override {
        printGLError();
        glBindTexture(GL_TEXTURE_2D,texture);
        int texSizeX = 256;
        int texSizeY = 126;
        float texCellPhysicalSize = cellPhysicalSize * gridSizeX/texSizeX;
        size_t imageSize = texSizeX*texSizeY*4* sizeof(unsigned char);
        unsigned char* image = (unsigned char*) malloc(imageSize);
        unsigned char* imageGPU = nullptr;
        HANDLE_ERROR(cudaMalloc(&imageGPU, imageSize ));
        HANDLE_ERROR(cudaMemset(imageGPU,255,imageSize));


        Fluid_2D_PCISPH_Impl::updateTextureImpl<<< numBlocks, numThreads >>>(particles,particleCount,texCellPhysicalSize,texSizeX,imageGPU);
        CHECK_CUDA_ERROR("u t");

        HANDLE_ERROR(cudaMemcpy(image,imageGPU,imageSize, cudaMemcpyDeviceToHost));

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texSizeX, texSizeY, 0, GL_RGBA, GL_UNSIGNED_BYTE, image);
        glGenerateMipmap(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D,0);
        free(image);
        HANDLE_ERROR(cudaFree(imageGPU));
        printGLError();

    }



};

#endif //AQUARIUS_FLUID_2D_PCISPH_CUH
