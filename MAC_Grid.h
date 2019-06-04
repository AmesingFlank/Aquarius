//
// Created by AmesingFlank on 2019-04-16.
//

#ifndef AQUARIUS_MAC_GRID_H
#define AQUARIUS_MAC_GRID_H

#include <stdlib.h>
#include <memory>
#include "CudaCommons.h"
#include <cmath>

#define CONTENT_AIR  0.0
#define CONTENT_FLUID  1.0
#define CONTENT_SOLID  2.0

struct Cell2D{
    float pressure;
    float uX;
    float uY;
    float uX_new;
    float uY_new;
    //float signedDistance;
    float content;
    float content_new;
    int index;
    bool hasVelocityX = false;
    bool hasVelocityY = false;

    float fluid0Count = 0;
    float fluid1Count = 0;

};

struct InterpolationCoords2D{
    int x0 ;
    int x1 ;
    int y0 ;
    int y1 ;

    float u0 ;
    float u1 ;
    float v0 ;
    float v1 ;
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

    MAC_Grid_2D(int X,int Y,float cellPhysicalSize_):
            sizeX(X),sizeY(Y),cellPhysicalSize(cellPhysicalSize_),
            physicalSizeX(X*cellPhysicalSize),physicalSizeY(Y*cellPhysicalSize)
    {
        cells = createCells();
    }

    Cell2D** createCells(){
        int cellsCount = (sizeX+1)*(sizeY+1);
        int memorySize = sizeof(Cell2D)*cellsCount;
        std::cout<<"malloc size "<<memorySize<<std::endl;
        Cell2D* data = new Cell2D[cellsCount];
        std::memset(data,0, sizeof(Cell2D)*cellsCount);
        Cell2D** result = new Cell2D*[sizeX+1];
        for (int x = 0; x <sizeX+1 ; ++x) {
            int offset = x* (sizeY+1);
            result[x] = &data[offset];
        }
        std::cout<< & result[sizeX][sizeY] <<std::endl;
        std::cout<< & data[(sizeX+1)*(sizeY+1)-1] <<std::endl;
        return result;
    }


    float2 getCellVelocity(int x,int y){
        if(x < 0 ||x > sizeX-1 ||   y < 0|| y > sizeY-1  ){
            x = max(min(x,sizeX-1),0);
            y = max(min(y,sizeY-1),0);
        }
        float2 velocity = make_float2(cells[x][y].uX + cells[x+1][y].uX ,cells[x][y].uX + cells[1][y+1].uY);
        velocity /= 2.f;
        return velocity;
    }

    float2 getInterpolatedVelocity(float x, float y){
        x = max(min(x,sizeX-2.f),0.f);
        y = max(min(y,sizeY-2.f),0.f);
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

        float uX = weightX[0][0] * cells[i][j].uX +
                   weightX[1][0] * cells[i+1][j].uX +
                   weightX[0][1] * cells[i][j+1].uX +
                   weightX[1][1] * cells[i+1][j+1].uX;
        float uY = weightY[0][0] * cells[i][j].uY +
                   weightY[1][0] * cells[i+1][j].uY +
                   weightY[0][1] * cells[i][j+1].uY +
                   weightY[1][1] * cells[i+1][j+1].uY;
        return make_float2(uX,uY);
    }


    float2 getPointVelocity(float physicalX,float physicalY){
        float x = physicalX/cellPhysicalSize;
        float y = physicalY/cellPhysicalSize;

        float2 result;

        result.x = getInterpolatedVelocity(x,y-0.5).x;
        result.y = getInterpolatedVelocity(x-0.5,y).y;

        return result;

    }

    float getCellContent(int x,int y){
        if(x < 0 ||x > sizeX-1 ||   y < 0|| y > sizeY-1  ){
            x = max(min(x,sizeX-1),0);
            y = max(min(y,sizeY-1),0);
        }
        float content = cells[x][y].content;
        return content;
    }

    float getPointContent(float physicalX,float physicalY){
        float x = physicalX/cellPhysicalSize;
        float y = physicalY/cellPhysicalSize;

        int x0,x1,y0,y1;
        float u0,u1,v0,v1;

        if(x-floor(x) > 0.5f){
            x0 = floor(x);
        }
        else{
            x0 = floor(x)-1;
        }
        u0 = 1.5f + x0 - x;
        x1 = x0+1;
        u1 = 1.f-u0;

        if(y-floor(y) > 0.5f){
            y0 = floor(y);
        }
        else{
            y0 = floor(y)-1;
        }
        v0 = 1.5f + y0  - y;
        y1 = y0+1;
        v1 = 1.f-v0;

        float c00 = getCellContent(x0,y0);
        float c01 = getCellContent(x0,y1);
        float c10 = getCellContent(x1,y0);
        float c11 = getCellContent(x1,y1);

        float result = u0*v0*c00 + u0*v1*c01 + u1*v0*c10 + u1*v1*c11;
        if(result < 0.5){
            return CONTENT_AIR;
        }
        return CONTENT_FLUID;

    }


    float2 getPhysicalPos(int x,int y){
        return make_float2((x+0.5f)*cellPhysicalSize,(y+0.5f)*cellPhysicalSize);
    }


    void commitVelocityChanges(){
        for (int y = 0; y < sizeY+1 ; ++y) {
            for (int x = 0; x <sizeX+1 ; ++x) {
                cells[x][y].uY=cells[x][y].uY_new;
                cells[x][y].uX=cells[x][y].uX_new;
            }
        }
    }

    void commitContentChanges(){
        int index = 0;
        for (int y = 0; y < sizeY+1 ; ++y) {
            for (int x = 0; x <sizeX+1 ; ++x) {
                cells[x][y].content=cells[x][y].content_new;
                if(cells[x][y].content == CONTENT_FLUID){
                    cells[x][y].index = index;
                    index++;
                }
            }
        }
        fluidCount = index;
    }


};

#endif //AQUARIUS_MAC_GRID_H
