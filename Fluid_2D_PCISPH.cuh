//
// Created by AmesingFlank on 2019-07-01.
//

#ifndef AQUARIUS_FLUID_2D_PCISPH_CUH
#define AQUARIUS_FLUID_2D_PCISPH_CUH

#include "CudaCommons.h"
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include "WeightKernels.h"

__global__ struct Particle{
    float2 position;
    //float2 velocity;
};



__global__
void calcHashImpl(uint   *particleHashes,  // output
               Particle* particles,               // input: positions
               uint    particleCount,
               float cellPhysicalSize,int gridSizeX)
{
    uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= particleCount) return;

    Particle& p = particles[index];

    // get address in grid
    int x = p.position.x/cellPhysicalSize;
    int y = p.position.y/cellPhysicalSize;
    uint hash = y*gridSizeX + x;

    particleHashes[index] = hash;
}


__global__
void findCellStartEndImpl(uint   *particleHashes,
                  uint* cellStart, uint* cellEnd,
                  uint particleCount)
{
    uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= particleCount) return;

    int thisHash = particleHashes[index];
    if(index>0 && particleHashes[index-1]<thisHash){
        cellStart[thisHash] = index;
    }


    if(index<particleCount-1 && particleHashes[index+1]>thisHash){
        cellEnd[thisHash] = index;
    }
}



class Fluid_2D_PCISPH{
public:

    const uint gridSizeX = 256, gridSizeY = 128;
    const uint cellCount = gridSizeX*gridSizeY;
    const uint particleCount = cellCount*8;
    const float cellPhysicalSize = 10.f/(float)gridSizeY;

    uint * cellStart;
    uint * cellEnd;

    Particle* particles;
    uint* particleHashes;

    uint numThreads, numBlocks;


    Fluid_2D_PCISPH(){
        HANDLE_ERROR(cudaMalloc(&cellStart, cellCount* sizeof(*cellStart)));
        HANDLE_ERROR(cudaMalloc(&cellEnd, cellCount* sizeof(*cellEnd)));

        HANDLE_ERROR(cudaMalloc(&particleHashes, particleCount* sizeof(*particleHashes)));
        HANDLE_ERROR(cudaMalloc(&particles, particleCount* sizeof(*particles)));

        numThreads = min(256,particleCount);
        numBlocks = (particleCount % numThreads != 0) ?
                (particleCount / numThreads + 1) : (particleCount / numThreads);
        initFluid();

    }

    void performSpatialHashing(){
        calcHashImpl<<< numBlocks, numThreads >>>(particleHashes,particles,particleCount,cellPhysicalSize,gridSizeX);
        CHECK_CUDA_ERROR("calc hash");
        thrust::sort_by_key(thrust::device, particleHashes, particleHashes + particleCount, particles);
        findCellStartEndImpl<<< numBlocks, numThreads >>>(particleHashes,cellStart,cellEnd,particleCount);
        CHECK_CUDA_ERROR("find cell start end");

        std::cout<<"finished spatial hashing"<<std::endl;

    }

    void initFluid(){
        Particle* newParticles = new Particle[particleCount];
        for(int i = 0;i<particleCount;++i){
            newParticles[i].position.x = 0;
            newParticles[i].position.y = 0;
        }
        HANDLE_ERROR(cudaMemcpy(particles, newParticles, particleCount* sizeof(Particle), cudaMemcpyHostToDevice));
        delete []newParticles;
    }



};

#endif //AQUARIUS_FLUID_2D_PCISPH_CUH
