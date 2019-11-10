//
// Created by AmesingFlank on 2019-04-16.
//

#ifndef AQUARIUS_MAC_GRID_2D_CUH
#define AQUARIUS_MAC_GRID_2D_CUH

#include <stdlib.h>
#include <memory>
#include "GpuCommons.h"
#include <cmath>
#include "WeightKernels.h"
#include <thrust/functional.h>
#include <thrust/reduce.h>



#define CONTENT_AIR  0.0
#define CONTENT_FLUID  1.0
#define CONTENT_SOLID  2.0

#define get2D(arr,x,y) arr[(x)*(sizeY+1)+(y)]

__host__ __device__
struct Cell2D{
    float pressure;

    float2 velocity = make_float2(0,0);
    float2 newVelocity = make_float2(0,0);

    //float signedDistance;
    float content;
    float content_new;
    int fluidIndex;
    bool hasVelocityX = false;
    bool hasVelocityY = false;

    float fluid0Count = 0;
    float fluid1Count = 0;

	float divergence = 0;

};


namespace MAC_Grid_2D_Utils{

    __global__
    inline
    void writeContentsAndIndices(Cell2D *cells, int cellCount,
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
    void setFluidIndex(Cell2D *cells,int cellCount,  int* fluidCount) {
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
    void setContentToNewContent(Cell2D *cells, int cellCount) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= cellCount ) return;

        cells[index].content=cells[index].content_new;
    }


    __global__
    inline
    void writeSpeedX(Cell2D *cells, int cellCount, float* speedX) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= cellCount ) return;

        speedX[index] = abs(cells[index].velocity.x);
    }

    __global__
    inline
    void writeSpeedY(Cell2D *cells, int cellCount, float* speedY) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= cellCount ) return;

        speedY[index] = abs(cells[index].velocity.y);
    }

}




class MAC_Grid_2D{
public:
    const int sizeX;
    const int sizeY;
    const int cellCount;

    const float cellPhysicalSize;
    const float physicalSizeX;
    const float physicalSizeY;

    int numThreadsCell;
    int numBlocksCell;

    int fluidCount = 0;

    Cell2D* cells;

    MAC_Grid_2D(int X,int Y,float cellPhysicalSize_):
            sizeX(X),sizeY(Y),cellCount((X+1)*(Y+1)),cellPhysicalSize(cellPhysicalSize_),
            physicalSizeX(X*cellPhysicalSize),physicalSizeY(Y*cellPhysicalSize)
    {
        numThreadsCell = min(1024, cellCount);
        numBlocksCell = divUp(cellCount, numThreadsCell);
        createCells();
    }

    Cell2D* copyCellsToHost(){
        Cell2D* cellsTemp = new Cell2D[cellCount];
        HANDLE_ERROR(cudaMemcpy(cellsTemp,cells,cellCount*sizeof(Cell2D),cudaMemcpyDeviceToHost));
        return cellsTemp;
    }

    void copyCellsToDevice(Cell2D* cellsTemp){
        HANDLE_ERROR(cudaMemcpy(cells,cellsTemp,cellCount*sizeof(Cell2D),cudaMemcpyHostToDevice));
    }

    void createCells(){

        int memorySize = sizeof(Cell2D)*cellCount;
        std::cout<<"malloc size "<<memorySize<<std::endl;

        HANDLE_ERROR(cudaMalloc (&cells,memorySize));
        HANDLE_ERROR(cudaMemset(cells,0,memorySize));

    }

    __device__ __host__
    static inline float2 getCellVelocity(int x,int y,int sizeX, int sizeY, Cell2D* cells){
        if(x < 0 ||x > sizeX-1 ||   y < 0|| y > sizeY-1  ){
            x = max(min(x,sizeX-1),0);
            y = max(min(y,sizeY-1),0);
        }
        float2 velocity = make_float2(get2D(cells,x,y).velocity.x + get2D(cells,x+1,y).velocity.x ,get2D(cells,x,y).velocity.y + get2D(cells,x,y+1).velocity.y);
        velocity /= 2.f;
        return velocity;
    }

    __device__ __host__
    static inline float2 getInterpolatedVelocity(float x, float y,int sizeX,int sizeY,Cell2D* cells){
        x = max(min(x,sizeX-1.f),0.f);
        y = max(min(y,sizeY-1.f),0.f);
        int i = floor(x);
        int j = floor(y);

        float u[2];
        float v[2];
        float weight[2][2];


        u[0] = i + 1.f -x ;
        u[1] = 1.f - u[0];
        v[0] = j + 1.f -y ;
        v[1] = 1.f - v[0];

        for (int a = 0; a < 2 ; ++a) {
            for (int b = 0; b < 2 ; ++b) {
                weight[a][b] = u[a]*v[b];
            }
        }

        float2 result = weight[0][0] * get2D(cells,i,j).velocity +
                   weight[1][0] * get2D(cells,i+1,j).velocity +
                   weight[0][1] * get2D(cells,i,j+1).velocity +
                   weight[1][1] * get2D(cells,i+1,j+1).velocity;

        return result;
    }


    __device__ __host__
    static inline float2 getPointVelocity(float2 physicalPos, float cellPhysicalSize,int sizeX,int sizeY,Cell2D* cells){
        float x = physicalPos.x/cellPhysicalSize;
        float y = physicalPos.y/cellPhysicalSize;

        float2 result;

        result.x = getInterpolatedVelocity(x,y-0.5,sizeX,sizeY,cells).x;
        result.y = getInterpolatedVelocity(x-0.5,y,sizeX,sizeY,cells).y;

        return result;
    }



    __device__ __host__
    static inline float2 getInterpolatedNewVelocity(float x, float y,int sizeX,int sizeY,Cell2D* cells){
        x = max(min(x,sizeX-1.f),0.f);
        y = max(min(y,sizeY-1.f),0.f);
        int i = floor(x);
        int j = floor(y);

        float u[2];
        float v[2];
        float weight[2][2];


        u[0] = i + 1.f -x ;
        u[1] = 1.f - u[0];
        v[0] = j + 1.f -y ;
        v[1] = 1.f - v[0];

        for (int a = 0; a < 2 ; ++a) {
            for (int b = 0; b < 2 ; ++b) {
                weight[a][b] = u[a]*v[b];
            }
        }

        float2 result = weight[0][0] * get2D(cells,i,j).newVelocity +
                   weight[1][0] * get2D(cells,i+1,j).newVelocity +
                   weight[0][1] * get2D(cells,i,j+1).newVelocity +
                   weight[1][1] * get2D(cells,i+1,j+1).newVelocity;
        return result;
    }


    __device__ __host__
    static inline float2 getPointNewVelocity(float2 physicalPos, float cellPhysicalSize,int sizeX,int sizeY,Cell2D* cells){
        float x = physicalPos.x/cellPhysicalSize;
        float y = physicalPos.y/cellPhysicalSize;

        float2 result;

        result.x = getInterpolatedNewVelocity(x,y-0.5,sizeX,sizeY,cells).x;
        result.y = getInterpolatedNewVelocity(x-0.5,y,sizeX,sizeY,cells).y;

        return result;
    }




    __device__ __host__
    static inline float2 getPhysicalPos(int x,int y,float cellPhysicalSize){
        return make_float2((x+0.5f)*cellPhysicalSize,(y+0.5f)*cellPhysicalSize);
    }


    void commitContentChanges(){

        MAC_Grid_2D_Utils::setContentToNewContent<<<numBlocksCell,numThreadsCell>>>(cells,cellCount);
        updateFluidCount();
    }

    void updateFluidCount2(){
        Cell2D* cellsTemp = copyCellsToHost();
        int index = 0;

        for(int c = 0;c<cellCount;++c){
            Cell2D& thisCell = cellsTemp[c];
            if(thisCell.content==CONTENT_FLUID){
                thisCell.fluidIndex = index;
                index++;
            }
        }

        fluidCount = index;
        copyCellsToDevice(cellsTemp);
        delete []cellsTemp;
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

        MAC_Grid_2D_Utils::writeContentsAndIndices<<<numBlocksCell,numThreadsCell>>>(cells,cellCount,contentCopy0,contentCopy1,indices);
        CHECK_CUDA_ERROR("write contents and indices");

        thrust::stable_sort_by_key(thrust::device, contentCopy0, contentCopy0 + cellCount, cells);
        thrust::stable_sort_by_key(thrust::device, contentCopy1, contentCopy1 + cellCount, indices);

        fluidCount = cellCount;

        int* fluidCountDevice;
        HANDLE_ERROR(cudaMalloc (&fluidCountDevice,sizeof(*fluidCountDevice)));

        MAC_Grid_2D_Utils::setFluidIndex<<<numBlocksCell,numThreadsCell>>>(cells,cellCount,fluidCountDevice);
        CHECK_CUDA_ERROR("set fluid index");

        HANDLE_ERROR(cudaMemcpy(&fluidCount,fluidCountDevice,sizeof(fluidCount),cudaMemcpyDeviceToHost));

        thrust::stable_sort_by_key(thrust::device, indices, indices + cellCount, cells);

        HANDLE_ERROR(cudaFree(fluidCountDevice));
        HANDLE_ERROR(cudaFree(contentCopy0));
        HANDLE_ERROR(cudaFree(contentCopy1));
        HANDLE_ERROR(cudaFree(indices));

    }


    float getMaxSpeed2(){
        float maxSpeed = 0;

        Cell2D *cellsTemp = copyCellsToHost();

        for (int c = 0; c < (sizeY + 1) * (sizeX + 1); ++c) {
            Cell2D &thisCell = cellsTemp[c];

            if (thisCell.hasVelocityX) {
                maxSpeed = max(maxSpeed,  abs(thisCell.newVelocity.x));
            }
            if (thisCell.hasVelocityY) {
                maxSpeed = max(maxSpeed, abs(thisCell.newVelocity.y));
            }
        }
        delete[] cellsTemp;

        //maxSpeed = thrust::reduce(grid.cells, grid.cells+grid.cellCount,0,MAC_Grid_2D_Utils::GreaterCellSpeed());

        maxSpeed = maxSpeed*sqrt(2);
        return maxSpeed;
    }

    float getMaxSpeed(){
        float * speedX;
        float * speedY;

        HANDLE_ERROR(cudaMalloc (&speedX,cellCount*sizeof(*speedX)));
        HANDLE_ERROR(cudaMalloc (&speedY,cellCount*sizeof(*speedY)));

        MAC_Grid_2D_Utils::writeSpeedX<<<numBlocksCell,numThreadsCell>>>(cells,cellCount,speedX);
        CHECK_CUDA_ERROR("write vX");
        MAC_Grid_2D_Utils::writeSpeedY<<<numBlocksCell,numThreadsCell>>>(cells,cellCount,speedY);
        CHECK_CUDA_ERROR("write vY");

        float maxX = thrust::reduce(thrust::device,speedX, speedX+cellCount,0,thrust::maximum<float>());
        float maxY = thrust::reduce(thrust::device,speedY, speedY+cellCount,0,thrust::maximum<float>());

        float maxSpeed = max(maxX,maxY) *sqrt(2);

        HANDLE_ERROR(cudaFree (speedX));
        HANDLE_ERROR(cudaFree (speedY));


        return maxSpeed;
    }

};

#endif //AQUARIUS_MAC_GRID_2D_CUH
