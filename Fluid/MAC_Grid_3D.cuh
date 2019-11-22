//
// Created by AmesingFlank on 2019-04-16.
//

#ifndef AQUARIUS_MAC_GRID_3D_CUH
#define AQUARIUS_MAC_GRID_3D_CUH

#include <stdlib.h>
#include <memory>
#include "../GpuCommons.h"
#include <cmath>
#include "WeightKernels.h"
#include <thrust/functional.h>


#define CONTENT_AIR  0.0
#define CONTENT_FLUID  1.0
#define CONTENT_SOLID  2.0

#define get3D(arr,x,y,z) arr[(x)*(sizeY+1)*(sizeZ+1)+(y)*(sizeZ+1)+(z)]

__host__ __device__
struct Cell3D{
    float pressure;

    float3 velocity = make_float3(0,0,0);
    float3 newVelocity = make_float3(0,0,0);

    //float signedDistance;
    float content;
    float content_new;
    int fluidIndex;
    bool hasVelocityX = false;
    bool hasVelocityY = false;
    bool hasVelocityZ = false;

    float fluid0Count = 0;
    float fluid1Count = 0;

	float divergence;

	float density;
};


namespace MAC_Grid_3D_Utils{

    __global__
    inline
    void writeContentsAndIndices(Cell3D *cells, int cellCount,
            int* contentCopy0, int* contentCopy1, int* indices) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= cellCount ) return;


        if(cells[index].content==CONTENT_FLUID){
            contentCopy0[index] = 0;
            contentCopy1[index] = 0;
        } else{
            contentCopy0[index] = 1;
            contentCopy1[index] = 1;
        }

        indices[index]=index;
    }

    __global__
    inline
    void setFluidIndex(Cell3D *cells,int cellCount,  int* fluidCount) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= cellCount ) return;


        if(cells[index].content==CONTENT_FLUID){
            cells[index].fluidIndex = index;
            if(index+1 < cellCount && cells[index+1].content!=CONTENT_FLUID){
                *fluidCount = index+1;
            }
        }
    }

    __global__
    inline
    void setContentToNewContent(Cell3D *cells, int cellCount) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= cellCount ) return;

        cells[index].content=cells[index].content_new;
    }


    __global__
    inline
    void writeSpeedX(Cell3D *cells, int cellCount, float* speedX) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= cellCount ) return;

        speedX[index] = abs(cells[index].velocity.x);
    }

    __global__
    inline
    void writeSpeedY(Cell3D *cells, int cellCount, float* speedY) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= cellCount ) return;

        speedY[index] = abs(cells[index].velocity.y);
    }

    __global__
    inline
    void writeSpeedZ(Cell3D *cells, int cellCount, float* speedZ) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= cellCount ) return;

        speedZ[index] = abs(cells[index].velocity.z);
    }

}




class MAC_Grid_3D{
public:
    const int sizeX;
    const int sizeY;
    const int sizeZ;
    const int cellCount;

    const float cellPhysicalSize;
    const float physicalSizeX;
    const float physicalSizeY;

    int numThreadsCell;
    int numBlocksCell;

    int fluidCount = 0;

    Cell3D* cells;

    MAC_Grid_3D(int X,int Y,int Z,float cellPhysicalSize_):
            sizeX(X),sizeY(Y),sizeZ(Z),cellCount((X+1)*(Y+1)*(Z+1)),cellPhysicalSize(cellPhysicalSize_),
            physicalSizeX(X*cellPhysicalSize),physicalSizeY(Y*cellPhysicalSize)
    {
        numThreadsCell = min(1024, cellCount);
        numBlocksCell = divUp(cellCount, numThreadsCell);
        createCells();
    }

    Cell3D* copyCellsToHost(){
        Cell3D* cellsTemp = new Cell3D[cellCount];
        HANDLE_ERROR(cudaMemcpy(cellsTemp,cells,cellCount*sizeof(Cell3D),cudaMemcpyDeviceToHost));
        return cellsTemp;
    }

    void copyCellsToDevice(Cell3D* cellsTemp){
        HANDLE_ERROR(cudaMemcpy(cells,cellsTemp,cellCount*sizeof(Cell3D),cudaMemcpyHostToDevice));
    }

    void createCells(){

        int memorySize = sizeof(Cell3D)*cellCount;
        std::cout<<"malloc size "<<memorySize<<std::endl;

        HANDLE_ERROR(cudaMalloc (&cells,memorySize));
        HANDLE_ERROR(cudaMemset(cells,0,memorySize));

    }

    __device__ __host__
    static float3 getCellVelocity(int x,int y,int z,int sizeX, int sizeY, int sizeZ,Cell3D* cells){
        if(x < 0 ||x > sizeX-1 ||   y < 0|| y > sizeY-1 || z<0 || z>sizeZ-1 ){
            x = max(min(x,sizeX-1),0);
            y = max(min(y,sizeY-1),0);
            z = max(min(z,sizeZ-1),0);
        }
        float3 velocity = make_float3(
                get3D(cells,x,y,z).velocity.x + get3D(cells,x+1,y,z).velocity.x ,
                get3D(cells,x,y,z).velocity.y + get3D(cells,x,y+1,z).velocity.y,
                get3D(cells,x,y,z).velocity.z + get3D(cells,x,y,z+1).velocity.z );
        velocity /= 2.f;
        return velocity;
    }

    __device__ __host__
    static float3 getInterpolatedVelocity(float x, float y,float z,int sizeX,int sizeY,int sizeZ,Cell3D* cells){
        x = max(min(x,sizeX-1.f),0.f);
        y = max(min(y,sizeY-1.f),0.f);
        z = max(min(z,sizeZ-1.f),0.f);

        int i = floor(x);
        int j = floor(y);
        int k = floor(z);

        float u[2];
        float v[2];
        float w[2];
        float weight[2][2][2];


        u[0] = i + 1.f -x ;
        u[1] = 1.f - u[0];
        v[0] = j + 1.f -y ;
        v[1] = 1.f - v[0];
        w[0] = k + 1.f -z ;
        w[1] = 1.f - w[0];

        for (int a = 0; a < 2 ; ++a) {
            for (int b = 0; b < 2 ; ++b) {
                for (int c = 0;c<2;++c){
                    weight[a][b][c] = u[a]*v[b]*w[c];
                }
            }
        }

        float3 result =  weight[0][0][0] * get3D(cells,i,j,k).velocity +
                        weight[1][0][0] * get3D(cells,i+1,j,k).velocity +
                        weight[0][1][0] * get3D(cells,i,j+1,k).velocity +
                        weight[1][1][0] * get3D(cells,i+1,j+1,k).velocity+
                        weight[0][0][1] * get3D(cells,i,j,k+1).velocity +
                        weight[1][0][1] * get3D(cells,i+1,j,k+1).velocity +
                        weight[0][1][1] * get3D(cells,i,j+1,k+1).velocity +
                        weight[1][1][1] * get3D(cells,i+1,j+1,k+1).velocity;
        
        return result;
    }


    __device__ __host__
    static float3 getPointVelocity(float3 physicalPos, float cellPhysicalSize,int sizeX,int sizeY,float sizeZ,Cell3D* cells){
        float x = physicalPos.x/cellPhysicalSize;
        float y = physicalPos.y/cellPhysicalSize;
        float z = physicalPos.z/cellPhysicalSize;

        float3 result;

        result.x = getInterpolatedVelocity(x,y-0.5,z-0.5,sizeX,sizeY,sizeZ,cells).x;
        result.y = getInterpolatedVelocity(x-0.5,y,z-0.5,sizeX,sizeY,sizeZ,cells).y;
        result.z = getInterpolatedVelocity(x-0.5,y-0.5,z,sizeX,sizeY,sizeZ,cells).z;

        return result;
    }



    __device__ __host__
    static float3 getInterpolatedNewVelocity(float x, float y,float z,int sizeX,int sizeY,int sizeZ,Cell3D* cells){
        x = max(min(x,sizeX-1.f),0.f);
        y = max(min(y,sizeY-1.f),0.f);
        z = max(min(z,sizeZ-1.f),0.f);

        int i = floor(x);
        int j = floor(y);
        int k = floor(z);

        float u[2];
        float v[2];
        float w[2];
        float weight[2][2][2];


        u[0] = i + 1.f -x ;
        u[1] = 1.f - u[0];
        v[0] = j + 1.f -y ;
        v[1] = 1.f - v[0];
        w[0] = k + 1.f -z ;
        w[1] = 1.f - w[0];

        for (int a = 0; a < 2 ; ++a) {
            for (int b = 0; b < 2 ; ++b) {
                for (int c = 0;c<2;++c){
                    weight[a][b][c] = u[a]*v[b]*w[c];
                }
            }
        }

        float3 result =  weight[0][0][0] * get3D(cells,i,j,k).newVelocity +
                        weight[1][0][0] * get3D(cells,i+1,j,k).newVelocity +
                        weight[0][1][0] * get3D(cells,i,j+1,k).newVelocity +
                        weight[1][1][0] * get3D(cells,i+1,j+1,k).newVelocity+
                        weight[0][0][1] * get3D(cells,i,j,k+1).newVelocity +
                        weight[1][0][1] * get3D(cells,i+1,j,k+1).newVelocity +
                        weight[0][1][1] * get3D(cells,i,j+1,k+1).newVelocity +
                        weight[1][1][1] * get3D(cells,i+1,j+1,k+1).newVelocity;

        return result;
    }


    __device__ __host__
    static float3 getPointNewVelocity(float3 physicalPos, float cellPhysicalSize,int sizeX,int sizeY,float sizeZ,Cell3D* cells){
        float x = physicalPos.x/cellPhysicalSize;
        float y = physicalPos.y/cellPhysicalSize;
        float z = physicalPos.z/cellPhysicalSize;

        float3 result;

        result.x = getInterpolatedNewVelocity(x,y-0.5,z-0.5,sizeX,sizeY,sizeZ,cells).x;
        result.y = getInterpolatedNewVelocity(x-0.5,y,z-0.5,sizeX,sizeY,sizeZ,cells).y;
        result.z = getInterpolatedNewVelocity(x-0.5,y-0.5,z,sizeX,sizeY,sizeZ,cells).z;

        return result;
    }




    __device__ __host__
    static float3 getPhysicalPos(int x,int y,int z,float cellPhysicalSize){
        return make_float3((x+0.5f)*cellPhysicalSize,(y+0.5f)*cellPhysicalSize,(z+0.5f)*cellPhysicalSize);
    }


    void commitContentChanges(){

        MAC_Grid_3D_Utils::setContentToNewContent<<<numBlocksCell,numThreadsCell>>>(cells,cellCount);
        updateFluidCount();
    }

    void updateFluidCount(){
        int* contentCopy0;
        int* contentCopy1;
        int* indices;
        HANDLE_ERROR(cudaMalloc (&contentCopy0,cellCount*sizeof(*contentCopy0)));
        HANDLE_ERROR(cudaMalloc (&contentCopy1,cellCount*sizeof(*contentCopy1)));
        HANDLE_ERROR(cudaMalloc (&indices,cellCount*sizeof(*indices)));

        int numThreadsCell = min(1024, cellCount);
        int numBlocksCell = divUp(cellCount, numThreadsCell);


        MAC_Grid_3D_Utils::writeContentsAndIndices<<<numBlocksCell,numThreadsCell>>>(cells,cellCount,contentCopy0,contentCopy1,indices);
        CHECK_CUDA_ERROR("write contents and indices");

        thrust::stable_sort_by_key(thrust::device, contentCopy0, contentCopy0 + cellCount, cells);
        thrust::stable_sort_by_key(thrust::device, contentCopy1, contentCopy1 + cellCount, indices);

        fluidCount = cellCount;

        int* fluidCountDevice;
        HANDLE_ERROR(cudaMalloc (&fluidCountDevice,sizeof(*fluidCountDevice)));

        MAC_Grid_3D_Utils::setFluidIndex<<<numBlocksCell,numThreadsCell>>>(cells,cellCount,fluidCountDevice);
        CHECK_CUDA_ERROR("set fluid index");

        HANDLE_ERROR(cudaMemcpy(&fluidCount,fluidCountDevice,sizeof(fluidCount),cudaMemcpyDeviceToHost));

        thrust::stable_sort_by_key(thrust::device, indices, indices + cellCount, cells);

        HANDLE_ERROR(cudaFree(fluidCountDevice));
        HANDLE_ERROR(cudaFree(contentCopy0));
        HANDLE_ERROR(cudaFree(contentCopy1));
        HANDLE_ERROR(cudaFree(indices));

    }



    float getMaxSpeed(){
        float * speedX;
        float * speedY;
        float * speedZ;

        HANDLE_ERROR(cudaMalloc (&speedX,cellCount*sizeof(*speedX)));
        HANDLE_ERROR(cudaMalloc (&speedY,cellCount*sizeof(*speedY)));
        HANDLE_ERROR(cudaMalloc (&speedZ,cellCount*sizeof(*speedZ)));

        MAC_Grid_3D_Utils::writeSpeedX<<<numBlocksCell,numThreadsCell>>>(cells,cellCount,speedX);
        CHECK_CUDA_ERROR("write vX");
        MAC_Grid_3D_Utils::writeSpeedY<<<numBlocksCell,numThreadsCell>>>(cells,cellCount,speedY);
        CHECK_CUDA_ERROR("write vY");
        MAC_Grid_3D_Utils::writeSpeedZ<<<numBlocksCell,numThreadsCell>>>(cells,cellCount,speedZ);
        CHECK_CUDA_ERROR("write vZ");

        float maxX = thrust::reduce(thrust::device,speedX, speedX+cellCount,0,thrust::maximum<float>());
        float maxY = thrust::reduce(thrust::device,speedY, speedY+cellCount,0,thrust::maximum<float>());
        float maxZ = thrust::reduce(thrust::device,speedZ, speedZ+cellCount,0,thrust::maximum<float>());

        float maxSpeed = max(max(maxX,maxY),maxZ) *sqrt(3);

        HANDLE_ERROR(cudaFree (speedX));
        HANDLE_ERROR(cudaFree (speedY));
        HANDLE_ERROR(cudaFree (speedZ));

        return maxSpeed;
    }

};

#endif //AQUARIUS_MAC_GRID_3D_CUH
