//
// Created by AmesingFlank on 2019-04-16.
//

#ifndef AQUARIUS_MAC_GRID_2D_CUH
#define AQUARIUS_MAC_GRID_2D_CUH

#include <stdlib.h>
#include <memory>
#include "CudaCommons.h"
#include <cmath>

#define CONTENT_AIR  0.0
#define CONTENT_FLUID  1.0
#define CONTENT_SOLID  2.0


__host__ __device__
struct Cell2D{
    float pressure;

    float2 velocity;
    float2 newVelocity;

    //float signedDistance;
    float content;
    float content_new;
    int index;
    bool hasVelocityX = false;
    bool hasVelocityY = false;

    float fluid0Count = 0;
    float fluid1Count = 0;

};





class MAC_Grid_2D{
public:
    const int sizeX;
    const int sizeY;

    const float cellPhysicalSize;
    const float physicalSizeX;
    const float physicalSizeY;

    int fluidCount = 0;

    Cell2D** cells;
    Cell2D* cellsData;

    MAC_Grid_2D(int X,int Y,float cellPhysicalSize_):
            sizeX(X),sizeY(Y),cellPhysicalSize(cellPhysicalSize_),
            physicalSizeX(X*cellPhysicalSize),physicalSizeY(Y*cellPhysicalSize)
    {
        cells = createCells();
    }

    Cell2D* copyCellsToHost(){
        Cell2D* cellsTemp = new Cell2D[(sizeX+1)*(sizeY+1)];
        HANDLE_ERROR(cudaMemcpy(cellsTemp,cellsData,(sizeX+1)*(sizeY+1)*sizeof(Cell2D),cudaMemcpyDeviceToHost));
        return cellsTemp;
    }

    void copyCellsToDevice(Cell2D* cellsTemp){
        HANDLE_ERROR(cudaMemcpy(cellsData,cellsTemp,(sizeX+1)*(sizeY+1)*sizeof(Cell2D),cudaMemcpyHostToDevice));
    }

    Cell2D** createCells(){
        int cellsCount = (sizeX+1)*(sizeY+1);
        int memorySize = sizeof(Cell2D)*cellsCount;
        std::cout<<"malloc size "<<memorySize<<std::endl;

        HANDLE_ERROR(cudaMalloc (&cellsData,memorySize));
        HANDLE_ERROR(cudaMemset(cellsData,0,memorySize));

        Cell2D** result;
        HANDLE_ERROR(cudaMalloc (&result,(sizeX+1)*sizeof(Cell2D*)  ));

        Cell2D** resultHost = new Cell2D*[sizeX+1];

        for (int x = 0; x <sizeX+1 ; ++x) {
            int offset = x* (sizeY+1);
            resultHost[x] = cellsData + offset;
        }

        HANDLE_ERROR(cudaMemcpy(result,resultHost,(sizeX+1)*sizeof(Cell2D*),cudaMemcpyHostToDevice));

        delete []resultHost;

        return result;
    }

    __device__ __host__
    static float2 getCellVelocity(int x,int y,int sizeX, int sizeY, Cell2D** cells){
        if(x < 0 ||x > sizeX-1 ||   y < 0|| y > sizeY-1  ){
            x = max(min(x,sizeX-1),0);
            y = max(min(y,sizeY-1),0);
        }
        float2 velocity = make_float2(cells[x][y].velocity.x + cells[x+1][y].velocity.x ,cells[x][y].velocity.x + cells[1][y+1].velocity.y);
        velocity /= 2.f;
        return velocity;
    }

    __device__ __host__
    static float2 getInterpolatedVelocity(float x, float y,int sizeX,int sizeY,Cell2D** cells){
        x = max(min(x,sizeX-1.f),0.f);
        y = max(min(y,sizeY-1.f),0.f);
        int i = floor(x);
        int j = floor(y);

        float u[2];
        float v[2];
        float weightX[2][2];
        float weightY[2][2];


        u[0] = i + 1.f -x ;
        u[1] = 1.f - u[0];
        v[0] = j + 1.f -y ;
        v[1] = 1.f - v[0];

        for (int a = 0; a < 2 ; ++a) {
            for (int b = 0; b < 2 ; ++b) {
                weightX[a][b] = u[a]*v[b];
                weightY[a][b] = u[a]*v[b];
            }
        }

        float uX = weightX[0][0] * cells[i][j].velocity.x +
                   weightX[1][0] * cells[i+1][j].velocity.x +
                   weightX[0][1] * cells[i][j+1].velocity.x +
                   weightX[1][1] * cells[i+1][j+1].velocity.x;
        float uY = weightY[0][0] * cells[i][j].velocity.y +
                   weightY[1][0] * cells[i+1][j].velocity.y +
                   weightY[0][1] * cells[i][j+1].velocity.y +
                   weightY[1][1] * cells[i+1][j+1].velocity.y;
        return make_float2(uX,uY);
    }


    __device__ __host__
    static float2 getPointVelocity(float physicalX,float physicalY, float cellPhysicalSize,int sizeX,int sizeY,Cell2D** cells){
        float x = physicalX/cellPhysicalSize;
        float y = physicalY/cellPhysicalSize;

        float2 result;

        result.x = getInterpolatedVelocity(x,y-0.5,sizeX,sizeY,cells).x;
        result.y = getInterpolatedVelocity(x-0.5,y,sizeX,sizeY,cells).y;

        return result;
    }





    __device__ __host__
    static float2 getPhysicalPos(int x,int y,float cellPhysicalSize){
        return make_float2((x+0.5f)*cellPhysicalSize,(y+0.5f)*cellPhysicalSize);
    }


    void commitContentChanges(){

        Cell2D* cellsTemp = copyCellsToHost();
        //HANDLE_ERROR(cudaMemcpy(cellsTemp,cells[0],(sizeY+1)*(sizeX+1)*sizeof(Cell2D),cudaMemcpyDeviceToHost));

        int index = 0;

        for(int c = 0;c<(sizeY+1)*(sizeX+1);++c){
            Cell2D& thisCell = cellsTemp[c];
            thisCell.content = thisCell.content_new;
            if(thisCell.content==CONTENT_FLUID){
                thisCell.index = index;
                index++;
            }
        }

        fluidCount = index;
        copyCellsToDevice(cellsTemp);
        delete []cellsTemp;
    }


};

#endif //AQUARIUS_MAC_GRID_2D_CUH
