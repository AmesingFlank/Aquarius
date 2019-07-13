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
    float2 position = make_float2(0,0);
    float2 velocity = make_float2(0,0);
    float2 newPosition = make_float2(0,0);
    float2 newVelocity = make_float2(0,0);

    float2 acceleration = make_float2(0,0);
    float pressure = 0;
    float2 pressureForce = make_float2(0,0);
    float2 otherForces = make_float2(0,0);
    float density = 0;
    float lastDensity = 0;

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

    //printf("p %d --> c %d\n",index,hash);
}

__global__
void findCellStartEndImpl(uint   *particleHashes,
                  int* cellStart, int* cellEnd,
                  uint particleCount)
{
    uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= particleCount) return;

    int thisHash = particleHashes[index];


    if(index==0 || particleHashes[index-1]<thisHash){
        cellStart[thisHash] = index;
    }

    if(index == particleCount-1 ||  particleHashes[index+1]>thisHash){
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

    particle.pressure = 0;
    particle.pressureForce = make_float2(0,0);

}



__global__
void calcPositionVelocity(Particle* particles,uint particleCount,
        float gridBoundaryX,float gridBoundaryY,float timeStep,float cellPhysicalSize){
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= particleCount) return;

    Particle& particle = particles[index];
    particle.acceleration = particle.pressureForce+particle.otherForces;
    float2 newVelocity = particle.velocity + timeStep *  particle.acceleration;
    particle.newVelocity = newVelocity;

    float2 meanVelocity = (newVelocity + particle.velocity) / 2.f;
    particle.newPosition = particle.position + meanVelocity*timeStep;
    
    particle.newPosition.x = min(gridBoundaryX,max(0.f,particle.newPosition.x));
    particle.newPosition.y = min(gridBoundaryY,max(0.f,particle.newPosition.y));


}

__global__
void commitPositionVelocity(Particle* particles,uint particleCount){
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= particleCount) return;

    Particle& particle = particles[index];
    particle.position = particle.newPosition;
    particle.velocity = particle.newVelocity;


}

__global__
void calcDensity(Particle* particles,uint particleCount,int* cellStart,int* cellEnd,
        uint gridSizeX, uint gridSizeY, float cellPhysicalSize
        ,float restDensity, float SCP){
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= particleCount) return;
    Particle& particle = particles[index];
    int cellX = particle.newPosition.x/cellPhysicalSize;
    int cellY = particle.newPosition.y/cellPhysicalSize;
    int cellID = cellY*gridSizeX + cellX;

    const float h = cellPhysicalSize;
    float density = restDensity*SCP;

    int start = cellStart[cellID];
    int end = cellEnd[cellID];

    //printf("%d\n",cellID);
    if(start>0 && end>0){
       // printf("%d\n",end-start);
    }

    int cellsToCheck[9];
    for (int r = 0; r < 3 ; ++r) {
        for (int c = 0; c < 3 ; ++c) {
            cellsToCheck[r*3+c] = cellID + (r-1)*gridSizeX + (c-1);
        }
    }

    uint cellCount = gridSizeX*gridSizeY;

    for (int thisCell : cellsToCheck) {
        if(thisCell>=0 && thisCell< cellCount){
            for(int j = cellStart[thisCell]; j<=cellEnd[thisCell];++j){
                if(j>=0 && j<particleCount && j!=index){
                    density += poly6(particle.newPosition-particles[j].newPosition,h);
                }
            }
        }
    }


    particle.density = density;
    if(density==0){
        printf("shit density is 0 %d\n");
    }

}

__global__
void calcPressure(Particle* particles,uint particleCount,int* cellStart,int* cellEnd,
                       uint gridSizeX, uint gridSizeY, float cellPhysicalSize,
                       float restDensity,float timeStep){
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= particleCount) return;
    Particle& particle = particles[index];
    int cellX = particle.newPosition.x/cellPhysicalSize;
    int cellY = particle.newPosition.y/cellPhysicalSize;
    int cellID = cellY*gridSizeX + cellX;

    const float h = cellPhysicalSize;

    float deltaDensity = particle.density - restDensity;

    int cellsToCheck[9];
    for (int r = 0; r < 3 ; ++r) {
        for (int c = 0; c < 3 ; ++c) {
            cellsToCheck[r*3+c] = cellID + (r-1)*gridSizeX + (c-1);
        }
    }

    uint cellCount = gridSizeX*gridSizeY;

    float beta = timeStep*timeStep * 2 / (restDensity*restDensity);

    float2 sumKernalGrad = make_float2(0,0);
    float sumDot = 0;

    for (int thisCell : cellsToCheck) {
        if(thisCell>=0 && thisCell< cellCount){
            for(int j = cellStart[thisCell]; j<=cellEnd[thisCell];++j){
                if(j>=0 && j<particleCount && j!=index){
                    float2 weight = spikey_grad(particle.newPosition-particles[j].newPosition,h);
                    sumKernalGrad += weight;
                    sumDot += dot(weight,weight);
                    //printf("%f\n",length(weight));
                }
            }
        }
    }

    float dotSumSumDot = dot(sumKernalGrad,sumKernalGrad)+sumDot;

    float coeff = 1.0/(beta*dotSumSumDot);
    if(coeff==coeff+1) coeff = 1;
    coeff = 0.08;
    particle.pressure += coeff*deltaDensity;
    //printf("%f\n",coeff);

}


__global__
void calcPressureForce(Particle* particles,uint particleCount,int* cellStart,int* cellEnd,
                  uint gridSizeX, uint gridSizeY, float cellPhysicalSize,float restDensity){
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= particleCount) return;
    Particle& particle = particles[index];
    int cellX = particle.newPosition.x/cellPhysicalSize;
    int cellY = particle.newPosition.y/cellPhysicalSize;
    int cellID = cellY*gridSizeX + cellX;

    const float h = cellPhysicalSize;
    float2 Fp = make_float2(0,0);

    int cellsToCheck[9];
    for (int r = 0; r < 3 ; ++r) {
        for (int c = 0; c < 3 ; ++c) {
            cellsToCheck[r*3+c] = cellID + (r-1)*gridSizeX + (c-1);
        }
    }

    uint cellCount = gridSizeX*gridSizeY;

    for (int thisCell : cellsToCheck) {
        if(thisCell>=0 && thisCell< cellCount){
            for(int j = cellStart[thisCell]; j<=cellEnd[thisCell];++j){
                if(j>=0 && j<particleCount && j!=index){
                    if(particle.newPosition.x == particles[j].newPosition.x){
                        if(particle.newPosition.y == particles[j].newPosition.y){
                            particle.newPosition.x+=0.001;
                            particle.newPosition.y+=0.001;
                        }
                    }
                    float2 weight = spikey_grad(particle.newPosition-particles[j].newPosition,h);
                    if(weight.x!=weight.x || weight.y!=weight.y){
                        printf("shit, weight is nan\n");
                    }
                    Fp -= weight * (particle.pressure+particles[j].pressure)/(restDensity*restDensity);
                    //printf("%f\n",weight.x);
                }
            }
        }
    }

    //printf("%f\n",Fp.x);
    particle.pressureForce = Fp*1;

    if(particle.newPosition.x == 0 && particle.pressureForce.x <= 0){
        particle.pressureForce.x = 0.00001;
    }
    if(particle.newPosition.y == 0 && particle.pressureForce.y <= 0){
        particle.pressureForce.y = 0.00001;
    }

}


__global__
void updateTextureImpl(Particle* particles,uint particleCount,float cellPhysicalSize,
        uint texSizeX,unsigned char* result){
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= particleCount) return;
    Particle& particle = particles[index];
    int cellX = particle.position.x/cellPhysicalSize;
    int cellY = particle.position.y/cellPhysicalSize;
    int cellID = cellY*texSizeX + cellX;

    unsigned char* base = result+ cellID*4;
    base[0] = 0;
    base[1] = 0;
    base[2] = 255;
    base[3] = 255;
}







class Fluid_2D_PCISPH:public Fluid_2D{
public:



    const float gridBoundaryX = 6.f;
    const float gridBoundaryY = 3.f;


    const float restDensity = 100;
    const float partilceRadius = sqrt(1/restDensity/M_PI);
    const float SCP = 1;
    const float kernalRadius = sqrt(4.0/(M_PI*restDensity*SCP));


    const float cellPhysicalSize = kernalRadius;

    const uint gridSizeX = ceil(gridBoundaryX/cellPhysicalSize);
    const uint gridSizeY = ceil(gridBoundaryY/cellPhysicalSize);
    const uint cellCount = gridSizeX*gridSizeY;
    uint particleCount = 0;


    int * cellStart;
    int * cellEnd;

    Particle* particles;
    uint* particleHashes;

    uint numThreads, numBlocks;


    Fluid_2D_PCISPH(){
        std::cout<<"kernal radius "<<kernalRadius<<std::endl;
        std::cout<<"particle radius "<<partilceRadius<<std::endl;

        std::cout<<gridSizeX<<std::endl<<gridSizeY<<std::endl;
        std::cout<<poly6(make_float2(0,0),kernalRadius)<<std::endl;

        initFluid();
    }

    void performSpatialHashing(){
        calcHashImpl<<< numBlocks, numThreads >>>(particleHashes,particles,particleCount,cellPhysicalSize,gridSizeX);
        CHECK_CUDA_ERROR("calc hash");
        thrust::sort_by_key(thrust::device, particleHashes, particleHashes + particleCount, particles);

        /*
        for (int i = 0; i <particleCount ; ++i) {
            uint thisHash = 0;
            HANDLE_ERROR(cudaMemcpy(&thisHash, particleHashes+i, sizeof(uint), cudaMemcpyDeviceToHost));
            std::cout<<"hash of "<<i<<" is "<<thisHash<<std::endl;
        }
         */

        HANDLE_ERROR(cudaMemset(cellStart,-1,cellCount*sizeof(*cellStart)));
        HANDLE_ERROR(cudaMemset(cellEnd,-1,cellCount*sizeof(*cellEnd)));
        findCellStartEndImpl<<< numBlocks, numThreads >>>(particleHashes,cellStart,cellEnd,particleCount);
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
        //std::cout<<"starting init fluid"<<std::endl;
        /*for (int x = 0; x < gridSizeX; ++x) {
            for (int y = 0; y < gridSizeY ; ++y) {
                float2 pos = make_float2( (x+0.5f)*cellPhysicalSize, (y+0.5f)*cellPhysicalSize );
                if( pow(x- 0.5*gridSizeX,2)+pow(y- 0.7*gridSizeY ,2) <= pow(0.2*gridSizeY,2) ){
                    //createParticles(allParticles,pos);
                }
                else if(y<0.5*gridSizeY && x<1.5*gridSizeX){
                    createParticles(allParticles,pos);
                }
            }
        }*/
        for(int x = 0;x<50;++x){
            for (int y = 0; y < 30 ; ++y) {
                float2 pos = make_float2( (x+0.5f)*partilceRadius, (y+0.5f)*partilceRadius );
                createParticles(allParticles,pos);
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
        numBlocks = (particleCount % numThreads != 0) ?
                    (particleCount / numThreads + 1) : (particleCount / numThreads);

        //performSpatialHashing();
        //calcDensity<<< numBlocks, numThreads >>>(particles,particleCount,cellStart,cellEnd,gridSizeX,gridSizeY,cellPhysicalSize,restDensity);
        //CHECK_CUDA_ERROR("calcDensity 0");
    }

    void simulationStep(float totalTime){
        updateTexture();
        float thisTimeStep = 0.01f;

        performSpatialHashing();
        calcOtherForces<<< numBlocks, numThreads >>>(particles,particleCount);
        CHECK_CUDA_ERROR("calcOtherForces");
        for (int i = 0 ; i < 15 ; ++i) {
            calcPositionVelocity<<< numBlocks, numThreads >>>(particles,particleCount,gridBoundaryX,gridBoundaryY,thisTimeStep,cellPhysicalSize);
            CHECK_CUDA_ERROR("calcPositionVelocity");

            calcDensity<<< numBlocks, numThreads >>>(particles,particleCount,cellStart,cellEnd,gridSizeX,gridSizeY,cellPhysicalSize,restDensity,SCP);
            CHECK_CUDA_ERROR("calcDensity");

            calcPressure<<< numBlocks, numThreads >>>(particles,particleCount,cellStart,cellEnd,gridSizeX,gridSizeY,cellPhysicalSize,restDensity,thisTimeStep);
            CHECK_CUDA_ERROR("calcPressure");

            calcPressureForce<<< numBlocks, numThreads >>>(particles,particleCount,cellStart,cellEnd,gridSizeX,gridSizeY,cellPhysicalSize,restDensity);
            CHECK_CUDA_ERROR("calcPressureForce");

        }
        calcPositionVelocity<<< numBlocks, numThreads >>>(particles,particleCount,gridBoundaryX,gridBoundaryY,thisTimeStep,cellPhysicalSize);
        commitPositionVelocity<<< numBlocks, numThreads >>>(particles,particleCount);
        CHECK_CUDA_ERROR("calcPositionVelocity");
        updateTexture();
    }


    virtual void updateTexture()override {
        printGLError();
        glBindTexture(GL_TEXTURE_2D,texture);
        uint texSizeX = 256;
        uint texSizeY = 126;
        float texCellPhysicalSize = cellPhysicalSize * gridSizeX/texSizeX;
        size_t imageSize = texSizeX*texSizeY*4* sizeof(unsigned char);
        unsigned char* image = (unsigned char*) malloc(imageSize);
        unsigned char* imageGPU = nullptr;
        HANDLE_ERROR(cudaMalloc(&imageGPU, imageSize ));
        HANDLE_ERROR(cudaMemset(imageGPU,255,imageSize));


        updateTextureImpl<<< numBlocks, numThreads >>>(particles,particleCount,texCellPhysicalSize,texSizeX,imageGPU);
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
