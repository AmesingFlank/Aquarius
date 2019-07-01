//
// Created by AmesingFlank on 2019-04-16.
//

#ifndef AQUARIUS_FLUID_2D_SEMILAGRANGE_H
#define AQUARIUS_FLUID_2D_SEMILAGRANGE_H

#include "MAC_Grid.h"
#include "PCG.h"
#include <vector>
#include <utility>
#include "CudaCommons.h"
#include <unordered_map>

struct PressureEquation{
    //std::vector<std::pair<int,float>> terms;
    std::unordered_map<int,float> terms_map;
    std::vector<std::pair<int,float>> terms_list;
    float RHS;
    int x;
    int y;
};

class Fluid_2D_SemiLagrange{
public:
    const int sizeX = 256;
    const int sizeY = 128;
    const float cellPhysicalSize = 10.f/(float)sizeY;
    const float gravitationalAcceleration = 9.8;
    const float density = 1;
    MAC_Grid_2D grid = MAC_Grid_2D(sizeX,sizeY,cellPhysicalSize);
    std::vector<float3> particles;

    GLuint texture;

    Fluid_2D_SemiLagrange(){
        init();
        initTexture();
    }

    void simulationStep(float totalTime){
        float thisTimeStep = 0.05f;

        extrapolateVelocity(thisTimeStep);

        advectVelocity(thisTimeStep);
        grid.commitVelocityChanges();

        applyForces(thisTimeStep);
        fixBoundary();
        solvePressure(thisTimeStep);

        for (int y = 0; y < sizeY; ++y) {
            for (int x = 0; x < sizeX; ++x) {
                if (grid.cells[x][y].content != CONTENT_FLUID)
                    continue;
                Cell2D thisCell = grid.cells[x][y];
                Cell2D rightCell = grid.cells[x+1][y];
                Cell2D upCell = grid.cells[x][y+1];
                float div = (thisCell.uY - upCell.uY + thisCell.uX - rightCell.uX);
                if(abs(div)>0.01)
                    std::cout<<x<<" "<<y<<"     divergence: "<<div<<std::endl;
            }
        }

        moveParticles(thisTimeStep);
        grid.commitContentChanges();

        //fixBoundary();
    }

    void init(){

        //set everything to air first
        for (int y = 0; y <=sizeY ; ++y) {
            for (int x = 0; x <=sizeX ; ++x) {
                grid.cells[x][y].uX = 0;
                grid.cells[x][y].uY = 0;
                grid.cells[x][y].content = CONTENT_AIR;
            }
        }
        fixBoundary();

        grid.fluidCount = 0;
        createSquareFluid();
        createSphereFluid(grid.fluidCount);
    }

    void advectVelocity(float timeStep){
        for (int y = 0; y <sizeY ; ++y) {
            for (int x = 0; x <sizeX ; ++x) {
                if(grid.cells[x][y].content==CONTENT_AIR) continue;
                float2 thisVelocity = grid.getCellVelocity(x,y);
                float2 thisPos = grid.getPhysicalPos(x,y);
                float2 midPos = thisPos - thisVelocity * 0.5f * timeStep;
                float2 midVelocity = grid.getPointVelocity(midPos.x,midPos.y);
                float2 sourcePos = thisPos - midVelocity*timeStep;
                float2 sourceVelocity = grid.getPointVelocity(sourcePos.x,sourcePos.y);
                grid.cells[x][y].uX_new = sourceVelocity.x;
                grid.cells[x][y].uY_new = sourceVelocity.y;
                if(y+1 <= sizeY && grid.cells[x][y+1].content == CONTENT_AIR){
                    grid.cells[x][y+1].uY_new = sourceVelocity.y;
                }
                if(x+1 <= sizeX && grid.cells[x+1][y].content == CONTENT_AIR){
                    grid.cells[x+1][y].uX_new = sourceVelocity.x;
                }
            }
        }

    }

    void moveParticles(float timeStep){

        for (int y = 0; y <sizeY ; ++y) {
            for (int x = 0; x <sizeX ; ++x) {
                grid.cells[x][y].content_new =CONTENT_AIR;
            }
        }

        for(float3& particle3:particles){
            float2 particle = make_float2(particle3.x,particle3.y);

            float2 thisVelocity = grid.getPointVelocity(particle.x,particle.y);
            float2 midPos = particle + thisVelocity * 0.5f * timeStep;
            float2 midVelocity = grid.getPointVelocity(midPos.x,midPos.y);
            float2 destPos = particle + midVelocity*timeStep;
            int destCellX = floor(destPos.x/cellPhysicalSize);
            int destCellY = floor(destPos.y/cellPhysicalSize);
            destCellX = max(min(destCellX,sizeX-1),0);
            destCellY = max(min(destCellY,sizeY-1),0);
            grid.cells[destCellX][destCellY].content_new = CONTENT_FLUID;
            particle = destPos;

            particle3.x = particle.x;
            particle3.y = particle.y;

            if(particle3.z>0){
                grid.cells[destCellX][destCellY].fluid1Count+=1;
            }
            else
                grid.cells[destCellX][destCellY].fluid0Count+=1;
        }
    }

    void applyForces(float timeStep){
        for (int y = 0; y <sizeY ; ++y) {
            for (int x = 0; x <sizeX ; ++x) {
                if(grid.cells[x][y].content==CONTENT_FLUID){
                    grid.cells[x][y].uY -= gravitationalAcceleration*timeStep;
                    if(grid.cells[x][y+1].content == CONTENT_AIR) grid.cells[x][y+1].uY -= gravitationalAcceleration*timeStep;
                }
                else if(grid.cells[x][y].content==CONTENT_AIR){
                    //if( x-1 >0 && grid.cells[x-1][y].content == CONTENT_AIR) grid.cells[x][y].uX = 0;
                    //if( y-1 >0 && grid.cells[x][y-1].content == CONTENT_AIR) grid.cells[x][y].uY = 0;
                }
            }
        }
    }

    void fixBoundary(){
        for (int y = 0; y <= sizeY ; ++y) {
            grid.cells[0][y].uX = 0;
            grid.cells[sizeX][y].uX = 0;
            grid.cells[0][y].hasVelocityX = true;
            grid.cells[sizeX][y].hasVelocityX = true;
            grid.cells[sizeX][y].content = CONTENT_SOLID;
        }
        for (int x = 0; x <= sizeX ; ++x) {
            grid.cells[x][0].uY = 0;
            //grid.cells[x][sizeY].uY = 0;
            grid.cells[x][0].hasVelocityY = true;
            grid.cells[x][sizeY].hasVelocityY = true;
            grid.cells[x][sizeY].content = CONTENT_AIR;
        }
    }

    float getNeibourCoefficient(int x,int y,float temp,float u,float&centerCoefficient,float& RHS){
        if (x >= 0 && x < sizeX && y>=0 && y<sizeY && grid.cells[x][y].content==CONTENT_FLUID){
            return temp*-1;
        }
        else{
            if (x < 0 || y<0 || x >= sizeX || grid.cells[x][y].content==CONTENT_SOLID){
                centerCoefficient -= temp;
                RHS += u;
                return 0;
            }
            else if ( y >= sizeY ||  grid.cells[x][y].content==CONTENT_AIR){
                return 0;
            }
            else{
                std::cout<<"This shoudld not happen"<<std::endl;
            }
        }
    }

    void solvePressure(float timeStep){

        std::vector<PressureEquation> equations;
        int nnz = 0;

        bool hasNonZeroRHS = false;

        float temp = timeStep/(density*cellPhysicalSize);

        for (int y = 0; y < sizeY; ++y) {
            for (int x = 0; x < sizeX; ++x) {
                grid.cells[x][y].pressure = 0;
                if(grid.cells[x][y].content!=CONTENT_FLUID)
                    continue;
                Cell2D thisCell = grid.cells[x][y];
                Cell2D rightCell = grid.cells[x+1][y];
                Cell2D upCell = grid.cells[x][y+1];

                PressureEquation thisEquation;
                float RHS = (thisCell.uY-upCell.uY + thisCell.uX-rightCell.uX);

                float centerCoeff = temp*4;

                float leftCoeff = getNeibourCoefficient(x-1,y,temp,thisCell.uX,centerCoeff,RHS);
                float rightCoeff = getNeibourCoefficient(x+1,y,temp,rightCell.uX,centerCoeff,RHS);
                float downCoeff = getNeibourCoefficient(x,y-1,temp,thisCell.uY,centerCoeff,RHS);
                float upCoeff = getNeibourCoefficient(x,y+1,temp,upCell.uY,centerCoeff,RHS);

                if(downCoeff){
                    Cell2D downCell = grid.cells[x][y-1];
                    thisEquation.terms_map[downCell.index]=downCoeff;
                    thisEquation.terms_list.push_back({downCell.index,downCoeff});
                    ++ nnz;
                }
                if(leftCoeff){
                    Cell2D leftCell = grid.cells[x-1][y];
                    thisEquation.terms_map[leftCell.index]=leftCoeff;
                    thisEquation.terms_list.push_back({leftCell.index,leftCoeff});
                    ++ nnz;
                }
                thisEquation.terms_map[thisCell.index]=centerCoeff;
                thisEquation.terms_list.push_back({thisCell.index,centerCoeff});
                if(rightCoeff){
                    thisEquation.terms_map[rightCell.index]=rightCoeff;
                    thisEquation.terms_list.push_back({rightCell.index,rightCoeff});
                    ++ nnz;
                }
                if(upCoeff){
                    thisEquation.terms_map[upCell.index]=upCoeff;
                    thisEquation.terms_list.push_back({upCell.index,upCoeff});
                    ++ nnz;
                }
                ++ nnz;
                thisEquation.RHS = RHS;
                if(RHS!=0){
                    hasNonZeroRHS = true;
                }
                thisEquation.x = x;
                thisEquation.y = y;
                equations.push_back(thisEquation);
            }
        }

        if(!hasNonZeroRHS){
            std::cout<<"zero RHS"<<std::endl;
            extrapolateVelocity(timeStep);
            return;
        }


        //number of rows == number of variables == number of fluid cells
        int n = equations.size();


        
        //construct the matrix of the linear equations
        int nnz_A = nnz;
        double* A_host = (double*) malloc(nnz_A* sizeof(*A_host));
        int* A_rowPtr_host = (int*) malloc( (n+1) * sizeof(*A_rowPtr_host));
        int* A_colInd_host = (int*) malloc( nnz_A * sizeof(*A_colInd_host));

        //construct a symmetric copy, used for computing the preconditioner
        int nnz_R = (nnz-n)/2 + n;
        nnz_R = n;
        double* R_host = (double*) malloc(nnz_R* sizeof(*R_host));
        int* R_rowPtr_host = (int*) malloc( (n+1) * sizeof(*R_rowPtr_host));
        int* R_colInd_host = (int*) malloc( nnz_R * sizeof(*R_colInd_host));

        for (int row = 0,i=0; row<n;++row) {
            PressureEquation& thisEquation = equations[row];
            A_rowPtr_host[row] = i;
            for (std::pair<int,float> coeffs: thisEquation.terms_list){
                //if(coeffs.first > row) continue;
                A_host[i] = coeffs.second;
                A_colInd_host[i]=coeffs.first;
                ++i;
            }
        }

        for (int row = 0,i=0; row<n;++row) {
            PressureEquation& thisEquation = equations[row];
            R_rowPtr_host[row] = i;
            for (std::pair<int,float> coeffs: thisEquation.terms_list){
                if(coeffs.first < row) continue;
                R_host[i] = coeffs.second;
                R_host[i] = 1; if(coeffs.first != row) continue;
                R_colInd_host[i]=coeffs.first;
                ++i;
            }
        }

        A_rowPtr_host[n] = nnz_A;
        R_rowPtr_host[n] = nnz_R;

        double *A_device;
        HANDLE_ERROR(cudaMalloc(&A_device, nnz_A * sizeof(*A_device)));
        HANDLE_ERROR(cudaMemcpy(A_device, A_host, nnz_A * sizeof(*A_device), cudaMemcpyHostToDevice));

        int *A_rowPtr_device;
        HANDLE_ERROR(cudaMalloc(&A_rowPtr_device, (n + 1) * sizeof(*A_rowPtr_device)));
        HANDLE_ERROR(cudaMemcpy(A_rowPtr_device, A_rowPtr_host, (n+1) * sizeof(*A_rowPtr_device), cudaMemcpyHostToDevice));

        int *A_colInd_device;
        HANDLE_ERROR(cudaMalloc(&A_colInd_device, nnz_A * sizeof(*A_colInd_device)));
        HANDLE_ERROR(cudaMemcpy(A_colInd_device, A_colInd_host, nnz_A * sizeof(*A_colInd_device), cudaMemcpyHostToDevice));

        cusparseMatDescr_t descrA;
        HANDLE_ERROR(cusparseCreateMatDescr(&descrA));
        //cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_SYMMETRIC);
        //cusparseSetMatFillMode(descrA, CUSPARSE_FILL_MODE_LOWER);
        cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_NON_UNIT);
        cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
        
        SparseMatrixCSR A(n,n,A_device,A_rowPtr_device,A_colInd_device,descrA,nnz_A);

        double *R_device;
        HANDLE_ERROR(cudaMalloc(&R_device, nnz_R * sizeof(*R_device)));
        HANDLE_ERROR(cudaMemcpy(R_device, R_host, nnz_R * sizeof(*R_device), cudaMemcpyHostToDevice));

        int *R_rowPtr_device;
        HANDLE_ERROR(cudaMalloc(&R_rowPtr_device, (n + 1) * sizeof(*R_rowPtr_device)));
        HANDLE_ERROR(cudaMemcpy(R_rowPtr_device, R_rowPtr_host, (n+1) * sizeof(*R_rowPtr_device), cudaMemcpyHostToDevice));

        int *R_colInd_device;
        HANDLE_ERROR(cudaMalloc(&R_colInd_device, nnz_R * sizeof(*R_colInd_device)));
        HANDLE_ERROR(cudaMemcpy(R_colInd_device, R_colInd_host, nnz_R * sizeof(*R_colInd_device), cudaMemcpyHostToDevice));

        cusparseMatDescr_t descrR;
        HANDLE_ERROR(cusparseCreateMatDescr(&descrR));
        cusparseSetMatType(descrR, CUSPARSE_MATRIX_TYPE_SYMMETRIC);
        cusparseSetMatFillMode(descrR, CUSPARSE_FILL_MODE_UPPER);
        //cusparseSetMatType(descrR, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatDiagType(descrR, CUSPARSE_DIAG_TYPE_NON_UNIT);
        cusparseSetMatIndexBase(descrR, CUSPARSE_INDEX_BASE_ZERO);

        SparseMatrixCSR R(n,n,R_device,R_rowPtr_device,R_colInd_device,descrR,nnz_R);
/*
        cusparseSolveAnalysisInfo_t ic0Info = 0;

        HANDLE_ERROR(cusparseCreateSolveAnalysisInfo(&ic0Info));

        HANDLE_ERROR(cusparseDcsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                n,nnz_R, descrR, R_device, R_rowPtr_device, R_colInd_device, ic0Info));

        HANDLE_ERROR(cusparseDcsric0(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, descrR,
                                         R_device, R_rowPtr_device, R_colInd_device, ic0Info));

        HANDLE_ERROR(cusparseDestroySolveAnalysisInfo(ic0Info));
*/
        cusparseSetMatType(descrR, CUSPARSE_MATRIX_TYPE_TRIANGULAR);

        //RHS vector
        double* f_host = (double*) malloc(n* sizeof(*f_host));
        for (int i = 0; i < n ; ++i) {
            f_host[i] = equations[i].RHS;
        }

        //solve the pressure equation
        double *result_device = solveSPD4(A,R,f_host,n);

        double *result_host = new double[n];
        HANDLE_ERROR(cudaMemcpy(result_host, result_device, n * sizeof(*result_host), cudaMemcpyDeviceToHost));


        for (int i = 0; i < n ; ++i) {
            PressureEquation& thisEquation = equations[i];
            int x = thisEquation.x;
            int y = thisEquation.y;
            float thisResult = result_host[i];
            grid.cells[x][y].pressure = thisResult;
        }


        //update velocity

        for (int y = 0; y < sizeY ; ++y) {
            for (int x = 0; x < sizeX; ++x) {
                Cell2D& thisCell = grid.cells[x][y];

                if(x>0){
                    Cell2D leftCell = grid.cells[x-1][y];
                    if(thisCell.content==CONTENT_FLUID || leftCell.content==CONTENT_FLUID){
                        float uX = thisCell.uX - temp* (thisCell.pressure-leftCell.pressure);
                        thisCell.uX = uX;
                    }
                }
                if(y>0){
                    Cell2D downCell = grid.cells[x][y-1];
                    if(thisCell.content==CONTENT_FLUID || downCell.content==CONTENT_FLUID){
                        float uY = thisCell.uY - temp* (thisCell.pressure-downCell.pressure);
                        thisCell.uY = uY;
                    }
                }

            }
        }

        // Extrapolate velocity into air cells;
        extrapolateVelocity(timeStep);


        A.free();
        R.free();
        free(f_host);
        HANDLE_ERROR(cudaFree(result_device));
        delete[] (result_host);

    }

    void extrapolateVelocity(float timeStep){
        for (int y = 0; y < sizeY ; ++y) {
            for (int x = 0; x < sizeX; ++x) {
                Cell2D& thisCell = grid.cells[x][y];
                thisCell.hasVelocityX = false;
                thisCell.hasVelocityY = false;
            }
        }
        //used to decide how far to extrapolate
        float maxSpeed = 0;

        for (int y = 0; y < sizeY ; ++y) {
            for (int x = 0; x < sizeX; ++x) {
                Cell2D& thisCell = grid.cells[x][y];
                if(x>0){
                    Cell2D leftCell = grid.cells[x-1][y];
                    if(thisCell.content==CONTENT_FLUID || leftCell.content==CONTENT_FLUID){
                        float uX = thisCell.uX;
                        maxSpeed = max(maxSpeed,abs(uX)*2);
                        thisCell.hasVelocityX = true;
                    }
                }
                if(y>0){
                    Cell2D downCell = grid.cells[x][y-1];
                    if(thisCell.content==CONTENT_FLUID || downCell.content==CONTENT_FLUID){
                        float uY = thisCell.uY;
                        maxSpeed = max(maxSpeed,abs(uY)*2);
                        thisCell.hasVelocityY = true;
                    }
                }

            }
        }

        float maxDist = (maxSpeed*timeStep + 1)/cellPhysicalSize;
        for (int distance = 0; distance < maxDist  ; ++distance) {
            for (int y = 0; y < sizeY ; ++y) {
                for (int x = 0; x < sizeX; ++x) {
                    Cell2D& thisCell = grid.cells[x][y];
                    float uX = thisCell.uX;
                    float uY = thisCell.uY;
                    const float epsilon = 1e-6;
                    if(x>0){
                        Cell2D& leftCell = grid.cells[x-1][y];
                        if(leftCell.content!=CONTENT_FLUID && !leftCell.hasVelocityX && thisCell.hasVelocityX && uX < -epsilon) {
                            leftCell.uX = uX;
                            leftCell.hasVelocityX = true;
                        }
                    }
                    if(y>0){
                        Cell2D& downCell = grid.cells[x][y-1];
                        if(downCell.content!=CONTENT_FLUID &&! downCell.hasVelocityY && thisCell.hasVelocityY && uY < -epsilon) {
                            downCell.uY = uY;
                            downCell.hasVelocityY = true;
                        }
                    }
                    Cell2D& rightCell = grid.cells[x+1][y];
                    if(rightCell.content!=CONTENT_FLUID && thisCell.content!=CONTENT_FLUID && !rightCell.hasVelocityX && thisCell.hasVelocityX && uX>epsilon) {
                        rightCell.uX = thisCell.uX;
                        rightCell.hasVelocityX = true;
                    }
                    Cell2D& upCell = grid.cells[x][y+1];
                    if(upCell.content!=CONTENT_FLUID && thisCell.content!=CONTENT_FLUID &&! upCell.hasVelocityY && thisCell.hasVelocityY && uY > epsilon) {
                        upCell.uY = thisCell.uY;
                        upCell.hasVelocityY = true;
                    }
                }
            }
        }
    }


    void initTexture(){
        glGenTextures(1,&texture);
        glBindTexture(GL_TEXTURE_2D,texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    void updateTexture(){
        printGLError();
        glBindTexture(GL_TEXTURE_2D,texture);
        unsigned char* image = (unsigned char*) malloc(sizeX*sizeY*4);
        for (int y = 0; y <sizeY ; ++y) {
            for (int x = 0; x < sizeX ; ++x) {
                Cell2D& thisCell = grid.cells[x][y];
                unsigned char* base = image + 4* (sizeX*y + x);
                if(thisCell.content == CONTENT_FLUID){
                    float fluid1percentage = thisCell.fluid1Count/(thisCell.fluid1Count+thisCell.fluid0Count);
                    //fluid1percentage = 0;
                    base[0] = 255*fluid1percentage;
                    base[1] = 0;
                    base[2] = 255*(1-fluid1percentage);

                    thisCell.fluid1Count=thisCell.fluid0Count=0;
                } else{
                    base[0] = 255;
                    base[1] = 255;
                    base[2] = 255;
                }
                base[3] = 255;
            }
        }
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, sizeX, sizeY, 0, GL_RGBA, GL_UNSIGNED_BYTE, image);
        glGenerateMipmap(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D,0);
        free(image);
        printGLError();

    }

    void createParticles(float2 centerPos,float tag = 0){
        for (int particle = 0; particle < 8 ; ++particle) {
            float xBias = (random0to1()-0.5f)*cellPhysicalSize;
            float yBias = (random0to1()-0.5f)*cellPhysicalSize;
            float2 particlePos = centerPos+make_float2(xBias,yBias);
            particles.push_back(make_float3(particlePos.x,particlePos.y,tag));
        }
    }

    void createSquareFluid(int startIndex = 0){
        int index = startIndex;
        for (int y = 0 * sizeY; y < 0.2 * sizeY ; ++y) {
            for (int x = 0 * sizeX; x < 1 * sizeX ; ++x) {
                grid.cells[x][y].uX = 0;
                grid.cells[x][y].uY = 0;
                grid.cells[x][y].content = CONTENT_FLUID;
                grid.cells[x][y].index = index;
                ++index;
                float2 thisPos = grid.getPhysicalPos(x,y);
                createParticles(thisPos,0);
            }
        }


        grid.fluidCount = index;
    }

    void createSphereFluid(int startIndex = 0){
        int index = startIndex;
        for (int y = 0 * sizeY; y < 1 * sizeY ; ++y) {
            for (int x = 0 * sizeX; x < 1 * sizeX ; ++x) {
                if( pow(x- 0.5*sizeX,2)+pow(y- 0.7*sizeY ,2) <= pow(0.2*sizeY,2) ){
                    grid.cells[x][y].uX = 0;
                    grid.cells[x][y].uY = 0;
                    grid.cells[x][y].content = CONTENT_FLUID;
                    grid.cells[x][y].index = index;
                    ++index;

                    float2 thisPos = grid.getPhysicalPos(x,y);
                    createParticles(thisPos,1);
                }
            }
        }

        grid.fluidCount = index;
    }

};

#endif //AQUARIUS_FLUID_2D_SEMILAGRANGE_H
