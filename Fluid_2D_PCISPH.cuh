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

__host__ __device__
struct Particle{
    float2 position;
    float2 velocity;
    float2 acceleration;
    float pressure = 0;
    float2 pressureForce;
    float2 otherForces;
    float density;
    float lastDensity;

    __host__ __device__
    Particle(){

    }

    __host__ __device__
    Particle(float2 position_):position(position_){

    }
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

__global__
void calcOtherForces(Particle* particles, uint particleCount){
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= particleCount) return;

    Particle& particle = particles[index];
    particle.otherForces = make_float2(0,-9.8);

    particle.lastDensity = particle.density;

    //particle.pressure = 0;
    //particle.pressureForce = 0;

}



__global__
void calcPositionVelocity(Particle* particles,uint particleCount,
        float gridBoundaryX,float gridBoundaryY,float timeStep){
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= particleCount) return;

    Particle& particle = particles[index];
    particle.acceleration = particle.pressureForce+particle.otherForces;
    float2 newVelocity = particle.velocity + timeStep *  particle.acceleration;
    float2 meanVelocity = (newVelocity + particle.velocity) / 2.f;
    particle.position = particle.position + meanVelocity*timeStep;
    particle.position.x = min(gridBoundaryX,max(0.f,particle.position.x));
    particle.position.y = min(gridBoundaryY,max(0.f,particle.position.y));

}

__global__
void calcDensity(Particle* particles,uint particleCount,uint * cellStart,uint * cellEnd,
        uint gridSizeX, float cellPhysicalSize){
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= particleCount) return;
    Particle& particle = particles[index];
    int cellX = particle.position.x/cellPhysicalSize;
    int cellY = particle.position.y/cellPhysicalSize;
    int cellID = cellY*gridSizeX + cellX;
    float density = 0;
    const float h = cellPhysicalSize/2.f;
    for(uint j = cellStart[cellID]; j<=cellEnd[cellID];++j){
        density += poly6(particle.position-particles[j].position,h);
    }
    particle.density = density;
}

__global__
void calcPressure(Particle* particles,uint particleCount,uint * cellStart,uint * cellEnd,
                 uint gridSizeX, float cellPhysicalSize){
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= particleCount) return;
    Particle& particle = particles[index];
    int cellX = particle.position.x/cellPhysicalSize;
    int cellY = particle.position.y/cellPhysicalSize;
    int cellID = cellY*gridSizeX + cellX;

    float deltaDensity = particle.density - particle.lastDensity;
    particle.pressure += deltaDensity;

    const float h = cellPhysicalSize/2.f;
    float2 Fp = make_float2(0,0);
    for(uint j = cellStart[cellID]; j<=cellEnd[cellID];++j){
        Fp -= spikey_grad(particle.position-particles[j].position,h) *
                (particle.pressure+particles[j].pressure)/(2*particles[j].pressure);
    }
    particle.pressureForce = Fp;
}


__global__
void updateTextureImpl(Particle* particles,uint particleCount,float cellPhysicalSize,uint gridSizeX,unsigned char* result){
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= particleCount) return;
    Particle& particle = particles[index];
    int cellX = particle.position.x/cellPhysicalSize;
    int cellY = particle.position.y/cellPhysicalSize;
    int cellID = cellY*gridSizeX + cellX;

    result[cellID] = 255;
}







class Fluid_2D_PCISPH:public Fluid_2D{
public:

    const uint gridSizeX = 256, gridSizeY = 128;
    const uint cellCount = gridSizeX*gridSizeY;
    uint particleCount = 0;
    const float cellPhysicalSize = 10.f/(float)gridSizeY;

    uint * cellStart;
    uint * cellEnd;

    Particle* particles;
    uint* particleHashes;

    uint numThreads, numBlocks;


    Fluid_2D_PCISPH(){
        initFluid();

        HANDLE_ERROR(cudaMalloc(&cellStart, cellCount* sizeof(*cellStart)));
        HANDLE_ERROR(cudaMalloc(&cellEnd, cellCount* sizeof(*cellEnd)));

        HANDLE_ERROR(cudaMalloc(&particleHashes, particleCount* sizeof(*particleHashes)));

        numThreads = min(256,particleCount);
        numBlocks = (particleCount % numThreads != 0) ?
                (particleCount / numThreads + 1) : (particleCount / numThreads);

    }

    void performSpatialHashing(){
        calcHashImpl<<< numBlocks, numThreads >>>(particleHashes,particles,particleCount,cellPhysicalSize,gridSizeX);
        CHECK_CUDA_ERROR("calc hash");
        thrust::sort_by_key(thrust::device, particleHashes, particleHashes + particleCount, particles);
        findCellStartEndImpl<<< numBlocks, numThreads >>>(particleHashes,cellStart,cellEnd,particleCount);
        CHECK_CUDA_ERROR("find cell start end");

        std::cout<<"finished spatial hashing"<<std::endl;

    }

    void createParticles(std::vector<Particle>& particleList, float2 centerPos){
        for (int particle = 0; particle < 8 ; ++particle) {
            float xBias = (random0to1()-0.5f)*cellPhysicalSize;
            float yBias = (random0to1()-0.5f)*cellPhysicalSize;
            float2 particlePos = centerPos+make_float2(xBias,yBias);
            particleList.emplace_back(particlePos);
        }
    }

    void initFluid(){
        std::vector<Particle> allParticles;
        //std::cout<<"starting init fluid"<<std::endl;
        for (int x = 0; x < gridSizeX; ++x) {
            for (int y = 0; y < gridSizeY ; ++y) {
                float2 pos = make_float2( (x+0.5f)*cellPhysicalSize, (y+0.5f)*cellPhysicalSize );
                if( pow(x- 0.5*gridSizeX,2)+pow(y- 0.7*gridSizeY ,2) <= pow(0.2*gridSizeY,2) ){
                    createParticles(allParticles,pos);
                }
                else if(y<0.2*gridSizeY){
                    createParticles(allParticles,pos);
                }
            }
        }
        particleCount = allParticles.size();
        std::cout<<particleCount<<std::endl;


        Particle* newParticles = new Particle[particleCount];
        for(int i = 0;i<particleCount;++i){
            newParticles[i] = allParticles[i];
        }
        HANDLE_ERROR(cudaMalloc(&particles, particleCount* sizeof(*particles)));
        HANDLE_ERROR(cudaMemcpy(particles, newParticles, particleCount* sizeof(Particle), cudaMemcpyHostToDevice));
        delete []newParticles;
    }

    void simulationStep(float totalTime){
        float thisTimeStep = 0.05f;
        performSpatialHashing();
        calcOtherForces<<< numBlocks, numThreads >>>(particles,particleCount);
        for (int i = 0; i <200 ; ++i) {
            calcPositionVelocity<<< numBlocks, numThreads >>>
                (particles,particleCount,gridSizeX*cellPhysicalSize,gridSizeY*cellPhysicalSize,thisTimeStep);

            calcDensity<<< numBlocks, numThreads >>>
                (particles,particleCount,cellStart,cellEnd,gridSizeX,cellPhysicalSize);

            calcPressure<<< numBlocks, numThreads >>>
                (particles,particleCount,cellStart,cellEnd,gridSizeX,cellPhysicalSize);

        }
        calcPositionVelocity<<< numBlocks, numThreads >>>
             (particles,particleCount,gridSizeX*cellPhysicalSize,gridSizeY*cellPhysicalSize,thisTimeStep);
        updateTexture();
    }


    virtual void updateTexture()override {
        printGLError();
        glBindTexture(GL_TEXTURE_2D,texture);
        size_t imageSize = gridSizeX*gridSizeY*4;
        unsigned char* image = (unsigned char*) malloc(imageSize);
        unsigned char* imageGPU = nullptr;
        HANDLE_ERROR(cudaMalloc(&imageGPU, imageSize* sizeof(*imageGPU)));

        updateTextureImpl<<< numBlocks, numThreads >>>
            (particles,particleCount,cellPhysicalSize,gridSizeX,imageGPU);

        HANDLE_ERROR(cudaMemcpy(image,imageGPU,imageSize, cudaMemcpyDeviceToHost));

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, gridSizeX, gridSizeY, 0, GL_RGBA, GL_UNSIGNED_BYTE, image);
        glGenerateMipmap(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D,0);
        free(image);
        HANDLE_ERROR(cudaFree(imageGPU));
        printGLError();

    }



};

#endif //AQUARIUS_FLUID_2D_PCISPH_CUH
