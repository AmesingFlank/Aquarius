//
// Created by AmesingFlank on 2019-04-16.
//

#ifndef AQUARIUS_FLUID_2D_SEMILAGRANGE_CUH
#define AQUARIUS_FLUID_2D_SEMILAGRANGE_CUH

#include "MAC_Grid_2D.cuh"
#include "SPD_Solver.h"
#include <vector>
#include <utility>
#include "CudaCommons.h"
#include "Fluid_2D.h"
#include <unordered_map>


namespace Fluid_2D_SemiLagrange {

    __device__ __host__
    struct PressureEquation2D {
        //std::vector<std::pair<int,float>> terms;
        //std::unordered_map<int,float> terms_map;
        //std::vector<std::pair<int,float>> terms_list;

        int termsIndex[5];
        float termsCoeff[5];
        unsigned char termCount = 0;
        float RHS;
        int x;
        int y;
    };

    __device__ __host__
    struct Particle{
        float2 position;
        float kind = 0;
        Particle(){

        }
        Particle(float2 pos):position(pos){

        }
        Particle(float2 pos,float tag):position(pos),kind(tag){

        }
    };


    __global__
    void advectVelocityImpl(Cell2D **cells, int sizeX, int sizeY, float timeStep, float gravitationalAcceleration,
                            float cellPhysicalSize) {
        uint index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= sizeX * sizeY) return;

        int y = index / sizeX;
        int x = index - y * sizeX;

        if (cells[x][y].content == CONTENT_AIR) return;
        float2 thisVelocity = MAC_Grid_2D::getCellVelocity(x, y, sizeX, sizeY, cells);
        float2 thisPos = MAC_Grid_2D::getPhysicalPos(x, y, cellPhysicalSize);

        float2 u1 = MAC_Grid_2D::getPointVelocity(thisPos, cellPhysicalSize, sizeX, sizeY, cells);
        float2 u2 = MAC_Grid_2D::getPointVelocity(thisPos-timeStep*u1/2, cellPhysicalSize, sizeX, sizeY, cells);
        float2 u3 = MAC_Grid_2D::getPointVelocity(thisPos-timeStep*u2*3/4, cellPhysicalSize, sizeX, sizeY, cells);

        float2 sourcePos = thisPos - timeStep* ( u1*2/9 + u2*3/9 + u3*4/9 );


        float2 sourceVelocity =
                MAC_Grid_2D::getPointVelocity(sourcePos, cellPhysicalSize, sizeX, sizeY, cells);
        cells[x][y].newVelocity = sourceVelocity;
        if (y + 1 <= sizeY && cells[x][y + 1].content == CONTENT_AIR) {
            cells[x][y + 1].newVelocity.y = sourceVelocity.y;
        }
        if (x + 1 <= sizeX && cells[x + 1][y].content == CONTENT_AIR) {
            cells[x + 1][y].newVelocity.x = sourceVelocity.x;
        }
    }

    __global__
    void
    moveParticlesImpl(float timeStep, Cell2D **cells, Particle *particles, uint particleCount, int sizeX, int sizeY,
                      float cellPhysicalSize) {
        uint index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= particleCount) return;

        Particle &particle = particles[index];
        float2 beginPos = particle.position;



        float2 u1 = MAC_Grid_2D::getPointVelocity(beginPos, cellPhysicalSize, sizeX, sizeY, cells);
        float2 u2 = MAC_Grid_2D::getPointVelocity(beginPos+timeStep*u1/2, cellPhysicalSize, sizeX, sizeY, cells);
        float2 u3 = MAC_Grid_2D::getPointVelocity(beginPos+timeStep*u2*3/4, cellPhysicalSize, sizeX, sizeY, cells);

        float2 destPos = beginPos + timeStep* ( u1*2/9 + u2*3/9 + u3*4/9 );


        int destCellX = floor(destPos.x / cellPhysicalSize);
        int destCellY = floor(destPos.y / cellPhysicalSize);
        destCellX = max(min(destCellX, sizeX - 1), 0);
        destCellY = max(min(destCellY, sizeY - 1), 0);
        cells[destCellX][destCellY].content_new = CONTENT_FLUID;
        particle.position = destPos;


        if (particle.kind > 0) {
            cells[destCellX][destCellY].fluid1Count += 1;
        } else
            cells[destCellX][destCellY].fluid0Count += 1;
    }

    __global__
    void setAllContent(Cell2D **cells, int sizeX, int sizeY, float content) {
        uint index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= sizeX * sizeY) return;

        int y = index / sizeX;
        int x = index - y * sizeX;
        cells[x][y].content_new = content;
    }

    __global__
    void applyForcesImpl(Cell2D **cells, int sizeX, int sizeY, float timeStep, float gravitationalAcceleration) {
        uint index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= sizeX * sizeY) return;

        int y = index / sizeX;
        int x = index - y * sizeX;

        if (cells[x][y].content == CONTENT_FLUID) {
            cells[x][y].velocity.y -= gravitationalAcceleration * timeStep;
            if (cells[x][y + 1].content == CONTENT_AIR)
                cells[x][y + 1].velocity.y -= gravitationalAcceleration * timeStep;
        } else if (cells[x][y].content == CONTENT_AIR) {
            //if( x-1 >0 && grid.cells[x-1][y].content == CONTENT_AIR) grid.cells[x][y].velocity.x = 0;
            //if( y-1 >0 && grid.cells[x][y-1].content == CONTENT_AIR) grid.cells[x][y].velocity.y = 0;
        }

    }

    __global__
    void fixBoundaryX(Cell2D **cells, int sizeX, int sizeY) {
        uint index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index > sizeY) return;
        int y = index;

        cells[0][y].velocity.x = 0;
        cells[sizeX][y].velocity.x = 0;
        cells[0][y].hasVelocityX = true;
        cells[sizeX][y].hasVelocityX = true;
        cells[sizeX][y].content = CONTENT_SOLID;
    }

    __global__
    void fixBoundaryY(Cell2D **cells, int sizeX, int sizeY) {
        uint index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index > sizeX) return;
        int x = index;

        cells[x][0].velocity.y = 0;
        //cells[x][sizeY].velocity.y = 0;
        cells[x][0].hasVelocityY = true;
        cells[x][sizeY].hasVelocityY = true;
        cells[x][sizeY].content = CONTENT_AIR;
    }

    __device__ __host__

    float getNeibourCoefficient(int x, int y, float temp, float u, float &centerCoefficient, float &RHS, Cell2D **cells,
                                int sizeX, int sizeY) {
        if (x >= 0 && x < sizeX && y >= 0 && y < sizeY && cells[x][y].content == CONTENT_FLUID) {
            return temp * -1;
        } else {
            if (x < 0 || y < 0 || x >= sizeX || cells[x][y].content == CONTENT_SOLID) {
                centerCoefficient -= temp;
                RHS += u;
                return 0;
            } else if (y >= sizeY || cells[x][y].content == CONTENT_AIR) {
                return 0;
            }
        }
    }

    __global__
    void constructPressureEquations(Cell2D **cells, int sizeX, int sizeY, PressureEquation2D *equations, float temp,
                                    bool *hasNonZeroRHS) {

        uint index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= sizeX * sizeY) return;

        int y = index / sizeX;
        int x = index - y * sizeX;

        cells[x][y].pressure = 0;
        if (cells[x][y].content != CONTENT_FLUID)
            return;
        Cell2D &thisCell = cells[x][y];
        Cell2D &rightCell = cells[x + 1][y];
        Cell2D &upCell = cells[x][y + 1];

        PressureEquation2D thisEquation;
        float RHS = (thisCell.velocity.y - upCell.velocity.y + thisCell.velocity.x - rightCell.velocity.x);

        float centerCoeff = temp * 4;

        float leftCoeff = getNeibourCoefficient(x - 1, y, temp, thisCell.velocity.x, centerCoeff, RHS, cells, sizeX,
                                                sizeY);
        float rightCoeff = getNeibourCoefficient(x + 1, y, temp, rightCell.velocity.x, centerCoeff, RHS, cells, sizeX,
                                                 sizeY);
        float downCoeff = getNeibourCoefficient(x, y - 1, temp, thisCell.velocity.y, centerCoeff, RHS, cells, sizeX,
                                                sizeY);
        float upCoeff = getNeibourCoefficient(x, y + 1, temp, upCell.velocity.y, centerCoeff, RHS, cells, sizeX, sizeY);

        int nnz = 0;

        if (downCoeff) {
            Cell2D &downCell = cells[x][y - 1];
            thisEquation.termsIndex[thisEquation.termCount] = downCell.index;
            thisEquation.termsCoeff[thisEquation.termCount] = downCoeff;
            ++thisEquation.termCount;
            ++nnz;
        }
        if (leftCoeff) {
            Cell2D &leftCell = cells[x - 1][y];
            thisEquation.termsIndex[thisEquation.termCount] = leftCell.index;
            thisEquation.termsCoeff[thisEquation.termCount] = leftCoeff;
            ++thisEquation.termCount;
            ++nnz;
        }
        thisEquation.termsIndex[thisEquation.termCount] = thisCell.index;
        thisEquation.termsCoeff[thisEquation.termCount] = centerCoeff;
        ++thisEquation.termCount;
        if (rightCoeff) {
            thisEquation.termsIndex[thisEquation.termCount] = rightCell.index;
            thisEquation.termsCoeff[thisEquation.termCount] = rightCoeff;
            ++thisEquation.termCount;
            ++nnz;
        }
        if (upCoeff) {
            thisEquation.termsIndex[thisEquation.termCount] = upCell.index;
            thisEquation.termsCoeff[thisEquation.termCount] = upCoeff;
            ++thisEquation.termCount;
            ++nnz;
        }
        ++nnz;
        thisEquation.RHS = RHS;
        if (RHS != 0) {
            *hasNonZeroRHS = true;
        }
        thisEquation.x = x;
        thisEquation.y = y;
        equations[thisCell.index] = thisEquation;

    }

    __global__
    void setPressure(Cell2D **cells, int sizeX, int sizeY, double *pressureResult) {
        uint index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= sizeX * sizeY) return;

        int y = index / sizeX;
        int x = index - y * sizeX;

        if (cells[x][y].content != CONTENT_FLUID)
            return;

        cells[x][y].pressure = pressureResult[cells[x][y].index];
    }

    __global__
    void updateVelocityWithPressure(Cell2D **cells, int sizeX, int sizeY, float temp) {
        uint index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= sizeX * sizeY) return;

        int y = index / sizeX;
        int x = index - y * sizeX;

        Cell2D &thisCell = cells[x][y];

        thisCell.hasVelocityX = false;
        thisCell.hasVelocityY = false;

        if (x > 0) {
            Cell2D &leftCell = cells[x - 1][y];
            if (thisCell.content == CONTENT_FLUID || leftCell.content == CONTENT_FLUID) {
                float uX = thisCell.velocity.x - temp * (thisCell.pressure - leftCell.pressure);
                thisCell.velocity.x = uX;
                thisCell.hasVelocityX = true;
            }
        }
        if (y > 0) {
            Cell2D &downCell = cells[x][y - 1];
            if (thisCell.content == CONTENT_FLUID || downCell.content == CONTENT_FLUID) {
                float uY = thisCell.velocity.y - temp * (thisCell.pressure - downCell.pressure);
                thisCell.velocity.y = uY;
                thisCell.hasVelocityY = true;
            }
        }
    }

    __global__
    void extrapolateVelocityByOne(Cell2D **cells, int sizeX, int sizeY) {
        uint index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= sizeX * sizeY) return;

        int y = index / sizeX;
        int x = index - y * sizeX;

        Cell2D &thisCell = cells[x][y];
        const float epsilon = 1e-6;

        if(!thisCell.hasVelocityX){
            float sumNeighborX = 0;
            int neighborXCount = 0;
            if (x > 0) {
                Cell2D &leftCell = cells[x - 1][y];
                if (leftCell.hasVelocityX && leftCell.velocity.x > epsilon) {
                    sumNeighborX += leftCell.velocity.x;
                    neighborXCount++;
                }
            }
            if (y > 0) {
                Cell2D &downCell = cells[x][y-1];
                if (downCell.hasVelocityX && downCell.velocity.y > epsilon) {
                    sumNeighborX += downCell.velocity.x;
                    neighborXCount++;
                }
            }
            if (x < sizeX-1) {
                Cell2D &rightCell = cells[x + 1][y];
                if (rightCell.hasVelocityX && rightCell.velocity.x < -epsilon) {
                    sumNeighborX += rightCell.velocity.x;
                    neighborXCount++;
                }
            }
            if (y < sizeY-1) {
                Cell2D &upCell = cells[x][y + 1];
                if (upCell.hasVelocityX && upCell.velocity.y < -epsilon) {
                    sumNeighborX += upCell.velocity.x;
                    neighborXCount++;
                }
            }
            if (neighborXCount>0){
                thisCell.velocity.x = sumNeighborX/(float)neighborXCount;
                thisCell.hasVelocityX = true;
            }
        }

        if(!thisCell.hasVelocityY){
            float sumNeighborY = 0;
            int neighborYCount = 0;
            if (x > 0) {
                Cell2D &leftCell = cells[x - 1][y];
                if (leftCell.hasVelocityY && leftCell.velocity.x > epsilon) {
                    sumNeighborY += leftCell.velocity.y;
                    neighborYCount++;
                }
            }
            if (y > 0) {
                Cell2D &downCell = cells[x][y-1];
                if (downCell.hasVelocityY && downCell.velocity.y > epsilon) {
                    sumNeighborY += downCell.velocity.y;
                    neighborYCount++;
                }
            }
            if (x < sizeX-1) {
                Cell2D &rightCell = cells[x + 1][y];
                if (rightCell.hasVelocityY && rightCell.velocity.x < -epsilon) {
                    sumNeighborY += rightCell.velocity.y;
                    neighborYCount++;
                }
            }
            if (y < sizeY-1) {
                Cell2D &upCell = cells[x][y + 1];
                if (upCell.hasVelocityY && upCell.velocity.y < -epsilon) {
                    sumNeighborY += upCell.velocity.y;
                    neighborYCount++;
                }
            }
            if (neighborYCount > 0){
                thisCell.velocity.y = sumNeighborY / (float)neighborYCount;
                thisCell.hasVelocityY = true;
            }
        }


    }

    __global__
    void updateTextureImpl(Cell2D **cells, int sizeX, int sizeY, unsigned char *image) {
        uint index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= sizeX * sizeY) return;

        int y = index / sizeX;
        int x = index - y * sizeX;

        Cell2D &thisCell = cells[x][y];
        unsigned char *base = image + 4 * (sizeX * y + x);
        if (thisCell.content == CONTENT_FLUID) {
            float fluid1percentage = thisCell.fluid1Count / (thisCell.fluid1Count + thisCell.fluid0Count);
            //fluid1percentage = 0;
            base[0] = 255 * fluid1percentage;
            base[1] = 0;
            base[2] = 255 * (1 - fluid1percentage);

            thisCell.fluid1Count = thisCell.fluid0Count = 0;
        } else {
            //            if(thisCell.hasVelocityY && thisCell.hasVelocityX){
//                base[0] = 255;
//                base[1] = 255;
//                base[2] = 255;
//            } else{
//                base[0] = 0;
//                base[1] = 0;
//                base[2] = 0;
//            }
            base[0] = 255;
            base[1] = 255;
            base[2] = 255;
        }
        base[3] = 255;

    }


    __global__
    void commitVelocityChanges(Cell2D **cells, int sizeX, int sizeY) {
        uint index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= sizeX * sizeY) return;

        int y = index / sizeX;
        int x = index - y * sizeX;

        Cell2D &thisCell = cells[x][y];
        thisCell.velocity = thisCell.newVelocity;
    }

    class Fluid : public Fluid_2D {
    public:
        const int sizeX = 256;
        const int sizeY = 128;
        const float cellPhysicalSize = 10.f / (float) sizeY;
        const float gravitationalAcceleration = 9.8;
        const float density = 1;
        MAC_Grid_2D grid = MAC_Grid_2D(sizeX, sizeY, cellPhysicalSize);

        Particle *particles;
        int particleCount;

        uint numThreadsParticle, numBlocksParticle;
        uint numThreadsCell, numBlocksCell;

        Fluid() {
            init();
        }

        void simulationStep(float totalTime) {
            float thisTimeStep = 0.05f;

            extrapolateVelocity(thisTimeStep);

            advectVelocity(thisTimeStep);

            commitVelocityChanges << < numBlocksCell, numThreadsCell >> > (grid.cells, sizeX, sizeY);

            applyForces(thisTimeStep);
            fixBoundary();
            solvePressure(thisTimeStep);

            // Extrapolate velocity into air cells;
            extrapolateVelocity(thisTimeStep);

            /*
            for (int y = 0; y < sizeY; ++y) {
                for (int x = 0; x < sizeX; ++x) {
                    if (grid.cells[x][y].content != CONTENT_FLUID)
                        continue;
                    Cell2D thisCell = grid.cells[x][y];
                    Cell2D rightCell = grid.cells[x+1][y];
                    Cell2D upCell = grid.cells[x][y+1];
                    float div = (thisCell.velocity.y - upCell.velocity.y + thisCell.velocity.x - rightCell.velocity.x);
                    if(abs(div)>0.01)
                        std::cout<<x<<" "<<y<<"     divergence: "<<div<<std::endl;
                }
            }*/

            moveParticles(thisTimeStep);
            grid.commitContentChanges();

            //fixBoundary();
        }

        void init() {


            //set everything to air first

            Cell2D *cellsTemp = grid.copyCellsToHost();


            grid.fluidCount = 0;
            std::vector <Particle> particlesHost;
            createSquareFluid(particlesHost, cellsTemp);
            createSphereFluid(particlesHost, cellsTemp, grid.fluidCount);
            particleCount = particlesHost.size();

            grid.copyCellsToDevice(cellsTemp);
            delete []cellsTemp;

            HANDLE_ERROR(cudaMalloc(&particles, particleCount * sizeof(Particle)));
            Particle *particlesHostToCopy = new Particle[particleCount];
            for (int i = 0; i < particleCount; ++i) {
                particlesHostToCopy[i] = particlesHost[i];
            }
            HANDLE_ERROR(cudaMemcpy(particles, particlesHostToCopy, particleCount * sizeof(Particle),
                                    cudaMemcpyHostToDevice));
            delete[] particlesHostToCopy;

            numThreadsParticle = min(1024, particleCount);
            numBlocksParticle = divUp(particleCount, numThreadsParticle);

            numThreadsCell = min(1024, sizeX * sizeY);
            numBlocksCell = divUp(sizeX * sizeY, numThreadsCell);

            std::cout << numThreadsCell << std::endl << numBlocksCell << std::endl;

            fixBoundary();

        }

        void advectVelocity(float timeStep) {
            advectVelocityImpl << < numBlocksCell, numThreadsCell >> >
                                                   (grid.cells, sizeX, sizeY, timeStep, gravitationalAcceleration, cellPhysicalSize);
            cudaDeviceSynchronize();
            CHECK_CUDA_ERROR("advect velocity");

        }

        void moveParticles(float timeStep) {

            setAllContent << < numBlocksCell, numThreadsCell >> > (grid.cells, sizeX, sizeY, CONTENT_AIR);
            CHECK_CUDA_ERROR("set all to air");

            moveParticlesImpl << < numBlocksParticle, numThreadsParticle >> >
                                                      (timeStep, grid.cells, particles, particleCount, sizeX, sizeY, cellPhysicalSize);
            cudaDeviceSynchronize();
            CHECK_CUDA_ERROR("move particles");
            return;

        }

        void applyForces(float timeStep) {
            applyForcesImpl << < numBlocksCell, numThreadsCell >> >
                                                (grid.cells, sizeX, sizeY, timeStep, gravitationalAcceleration);
            cudaDeviceSynchronize();
            CHECK_CUDA_ERROR("apply forces");

        }

        void fixBoundary() {
            uint numThreads, numBlocks;

            numThreads = min(1024, sizeY);
            numBlocks = divUp(sizeY, numThreadsCell);
            fixBoundaryX << < numBlocks, numThreads >> > (grid.cells, sizeX, sizeY);
            cudaDeviceSynchronize();
            CHECK_CUDA_ERROR("fix boundary x");

            numThreads = min(1024, sizeX);
            numBlocks = divUp(sizeX, numThreadsCell);
            fixBoundaryY << < numBlocks, numThreads >> > (grid.cells, sizeX, sizeY);
            CHECK_CUDA_ERROR("fix boundary y");

        }


        void solvePressure(float timeStep) {


            PressureEquation2D *equations = new PressureEquation2D[grid.fluidCount];
            int nnz = 0;
            bool hasNonZeroRHS = false;
            float temp = timeStep / (density * cellPhysicalSize);


            PressureEquation2D *equationsDevice;
            HANDLE_ERROR(cudaMalloc(&equationsDevice, grid.fluidCount * sizeof(PressureEquation2D)));

            bool *hasNonZeroRHS_Device;
            HANDLE_ERROR(cudaMalloc(&hasNonZeroRHS_Device, sizeof(*hasNonZeroRHS_Device)));
            HANDLE_ERROR(cudaMemset(hasNonZeroRHS_Device,0,sizeof(*hasNonZeroRHS_Device)));

            constructPressureEquations << < numBlocksCell, numThreadsCell >> >
                                                           (grid.cells, sizeX, sizeY, equationsDevice, temp, hasNonZeroRHS_Device);
            cudaDeviceSynchronize();
            CHECK_CUDA_ERROR("construct eqns");


            HANDLE_ERROR(cudaMemcpy(equations, equationsDevice, grid.fluidCount * sizeof(PressureEquation2D),
                                    cudaMemcpyDeviceToHost));
            HANDLE_ERROR(
                    cudaMemcpy(&hasNonZeroRHS, hasNonZeroRHS_Device, sizeof(hasNonZeroRHS), cudaMemcpyDeviceToHost));

            HANDLE_ERROR(cudaFree(equationsDevice));
            HANDLE_ERROR(cudaFree(hasNonZeroRHS_Device));

            cudaDeviceSynchronize();

            for (int i = 0; i < grid.fluidCount; ++i) {
                nnz += equations[i].termCount;
            }


            if (!hasNonZeroRHS) {
                std::cout << "zero RHS" << std::endl;
                extrapolateVelocity(timeStep);
                return;
            }


            //number of rows == number of variables == number of fluid cells
            int numVariables = grid.fluidCount;



            //construct the matrix of the linear equations
            int nnz_A = nnz;
            double *A_host = (double *) malloc(nnz_A * sizeof(*A_host));
            int *A_rowPtr_host = (int *) malloc((numVariables + 1) * sizeof(*A_rowPtr_host));
            int *A_colInd_host = (int *) malloc(nnz_A * sizeof(*A_colInd_host));

            //construct a symmetric copy, used for computing the preconditioner
            int nnz_R = (nnz - numVariables) / 2 + numVariables;
            nnz_R = numVariables;
            double *R_host = (double *) malloc(nnz_R * sizeof(*R_host));
            int *R_rowPtr_host = (int *) malloc((numVariables + 1) * sizeof(*R_rowPtr_host));
            int *R_colInd_host = (int *) malloc(nnz_R * sizeof(*R_colInd_host));

            for (int row = 0, i = 0; row < numVariables; ++row) {
                PressureEquation2D &thisEquation = equations[row];
                A_rowPtr_host[row] = i;

                for (int term = 0; term < thisEquation.termCount; ++term) {
                    //if(thisEquation.termsIndex[term] > row) continue;
                    A_host[i] = thisEquation.termsCoeff[term];
                    A_colInd_host[i] = thisEquation.termsIndex[term];
                    ++i;
                }

            }

            for (int row = 0, i = 0; row < numVariables; ++row) {
                PressureEquation2D &thisEquation = equations[row];
                R_rowPtr_host[row] = i;
                for (int term = 0; term < thisEquation.termCount; ++term) {
                    if (thisEquation.termsIndex[term] < row) continue;
                    R_host[i] = thisEquation.termsCoeff[term];
                    R_host[i] = 1;
                    if (thisEquation.termsIndex[term] != row) continue;
                    R_colInd_host[i] = thisEquation.termsIndex[term];
                    ++i;
                }
            }

            A_rowPtr_host[numVariables] = nnz_A;
            R_rowPtr_host[numVariables] = nnz_R;

            double *A_device;
            HANDLE_ERROR(cudaMalloc(&A_device, nnz_A * sizeof(*A_device)));
            HANDLE_ERROR(cudaMemcpy(A_device, A_host, nnz_A * sizeof(*A_device), cudaMemcpyHostToDevice));

            int *A_rowPtr_device;
            HANDLE_ERROR(cudaMalloc(&A_rowPtr_device, (numVariables + 1) * sizeof(*A_rowPtr_device)));
            HANDLE_ERROR(cudaMemcpy(A_rowPtr_device, A_rowPtr_host, (numVariables + 1) * sizeof(*A_rowPtr_device),
                                    cudaMemcpyHostToDevice));

            int *A_colInd_device;
            HANDLE_ERROR(cudaMalloc(&A_colInd_device, nnz_A * sizeof(*A_colInd_device)));
            HANDLE_ERROR(cudaMemcpy(A_colInd_device, A_colInd_host, nnz_A * sizeof(*A_colInd_device),
                                    cudaMemcpyHostToDevice));

            cusparseMatDescr_t descrA;
            HANDLE_ERROR(cusparseCreateMatDescr(&descrA));
            //cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_SYMMETRIC);
            //cusparseSetMatFillMode(descrA, CUSPARSE_FILL_MODE_LOWER);
            cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
            cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_NON_UNIT);
            cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

            SparseMatrixCSR A(numVariables, numVariables, A_device, A_rowPtr_device, A_colInd_device, descrA, nnz_A);

            double *R_device;
            HANDLE_ERROR(cudaMalloc(&R_device, nnz_R * sizeof(*R_device)));
            HANDLE_ERROR(cudaMemcpy(R_device, R_host, nnz_R * sizeof(*R_device), cudaMemcpyHostToDevice));

            int *R_rowPtr_device;
            HANDLE_ERROR(cudaMalloc(&R_rowPtr_device, (numVariables + 1) * sizeof(*R_rowPtr_device)));
            HANDLE_ERROR(cudaMemcpy(R_rowPtr_device, R_rowPtr_host, (numVariables + 1) * sizeof(*R_rowPtr_device),
                                    cudaMemcpyHostToDevice));

            int *R_colInd_device;
            HANDLE_ERROR(cudaMalloc(&R_colInd_device, nnz_R * sizeof(*R_colInd_device)));
            HANDLE_ERROR(cudaMemcpy(R_colInd_device, R_colInd_host, nnz_R * sizeof(*R_colInd_device),
                                    cudaMemcpyHostToDevice));

            cusparseMatDescr_t descrR;
            HANDLE_ERROR(cusparseCreateMatDescr(&descrR));
            cusparseSetMatType(descrR, CUSPARSE_MATRIX_TYPE_SYMMETRIC);
            cusparseSetMatFillMode(descrR, CUSPARSE_FILL_MODE_UPPER);
            //cusparseSetMatType(descrR, CUSPARSE_MATRIX_TYPE_GENERAL);
            cusparseSetMatDiagType(descrR, CUSPARSE_DIAG_TYPE_NON_UNIT);
            cusparseSetMatIndexBase(descrR, CUSPARSE_INDEX_BASE_ZERO);

            SparseMatrixCSR R(numVariables, numVariables, R_device, R_rowPtr_device, R_colInd_device, descrR, nnz_R);
/*
        cusparseSolveAnalysisInfo_t ic0Info = 0;

        HANDLE_ERROR(cusparseCreateSolveAnalysisInfo(&ic0Info));

        HANDLE_ERROR(cusparseDcsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                numVariables,nnz_R, descrR, R_device, R_rowPtr_device, R_colInd_device, ic0Info));

        HANDLE_ERROR(cusparseDcsric0(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, numVariables, descrR,
                                         R_device, R_rowPtr_device, R_colInd_device, ic0Info));

        HANDLE_ERROR(cusparseDestroySolveAnalysisInfo(ic0Info));
*/
            cusparseSetMatType(descrR, CUSPARSE_MATRIX_TYPE_TRIANGULAR);

            //RHS vector
            double *f_host = (double *) malloc(numVariables * sizeof(*f_host));
            for (int i = 0; i < numVariables; ++i) {
                f_host[i] = equations[i].RHS;
            }

            //solve the pressure equation
            double *result_device = solveSPD3(A, R, f_host, numVariables);

            double *result_host = new double[numVariables];
            HANDLE_ERROR(cudaMemcpy(result_host, result_device, numVariables * sizeof(*result_host),
                                    cudaMemcpyDeviceToHost));


            setPressure << < numBlocksCell, numThreadsCell >> > (grid.cells, sizeX, sizeY, result_device);
            cudaDeviceSynchronize();
            CHECK_CUDA_ERROR("set pressure");


            //update velocity

            updateVelocityWithPressure << < numBlocksCell, numThreadsCell >> > (grid.cells, sizeX, sizeY, temp);
            cudaDeviceSynchronize();
            CHECK_CUDA_ERROR("update velocity with pressure");


            A.free();
            R.free();
            free(f_host);
            HANDLE_ERROR(cudaFree(result_device));
            delete[] (result_host);

            delete[] equations;

        }

        void extrapolateVelocity(float timeStep) {

            //used to decide how far to extrapolate
            float maxSpeed = 0;

            Cell2D *cellsTemp = grid.copyCellsToHost();

            for (int c = 0; c < (sizeY + 1) * (sizeX + 1); ++c) {
                Cell2D &thisCell = cellsTemp[c];
                if (thisCell.hasVelocityX) {
                    maxSpeed = max(maxSpeed, 2 * abs(thisCell.velocity.x));
                }
                if (thisCell.hasVelocityY) {
                    maxSpeed = max(maxSpeed, 2 * abs(thisCell.velocity.y));
                }
            }
            delete[] cellsTemp;

            float maxDist = (maxSpeed * timeStep + 1) / cellPhysicalSize;

            for (int distance = 0; distance < maxDist; ++distance) {
                extrapolateVelocityByOne << < numBlocksCell, numThreadsCell >> > (grid.cells, sizeX, sizeY);
                cudaDeviceSynchronize();
                CHECK_CUDA_ERROR("extrapolate vel");
            }
        }


        virtual void updateTexture() override {
            printGLError();
            glBindTexture(GL_TEXTURE_2D, texture);
            int imageMemorySize = sizeX * sizeY * 4;
            unsigned char *image = (unsigned char *) malloc(imageMemorySize);
            unsigned char *imageDevice;
            HANDLE_ERROR(cudaMalloc(&imageDevice, imageMemorySize));

            updateTextureImpl << < numBlocksCell, numThreadsCell >> > (grid.cells, sizeX, sizeY, imageDevice);
            cudaDeviceSynchronize();
            CHECK_CUDA_ERROR("update tex");

            HANDLE_ERROR(cudaMemcpy(image, imageDevice, imageMemorySize, cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaFree(imageDevice));

            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, sizeX, sizeY, 0, GL_RGBA, GL_UNSIGNED_BYTE, image);
            glGenerateMipmap(GL_TEXTURE_2D);
            glBindTexture(GL_TEXTURE_2D, 0);
            free(image);
            printGLError();

        }

        void createParticles(std::vector <Particle> &particlesHost, float2 centerPos, float tag = 0) {
            for (int particle = 0; particle < 8; ++particle) {
                float xBias = (random0to1() - 0.5f) * cellPhysicalSize;
                float yBias = (random0to1() - 0.5f) * cellPhysicalSize;
                float2 particlePos = centerPos + make_float2(xBias, yBias);
                particlesHost.emplace_back(particlePos,tag);
            }
        }

        void createSquareFluid(std::vector <Particle> &particlesHost, Cell2D *cellsTemp, int startIndex = 0) {
            int index = startIndex;
            for (int y = 0 * sizeY; y < 0.2 * sizeY; ++y) {
                for (int x = 0 * sizeX; x < 1 * sizeX; ++x) {

                    Cell2D &thisCell = cellsTemp[x * (sizeY + 1) + y];

                    thisCell.velocity.x = 0;
                    thisCell.velocity.y = 0;
                    thisCell.content = CONTENT_FLUID;
                    thisCell.index = index;
                    ++index;
                    float2 thisPos = MAC_Grid_2D::getPhysicalPos(x, y, cellPhysicalSize);
                    createParticles(particlesHost, thisPos, 0);
                }
            }

            grid.fluidCount = index;
        }

        void createSphereFluid(std::vector <Particle> &particlesHost, Cell2D *cellsTemp, int startIndex = 0) {
            int index = startIndex;
            for (int y = 0 * sizeY; y < 1 * sizeY; ++y) {
                for (int x = 0 * sizeX; x < 1 * sizeX; ++x) {
                    if (pow(x - 0.5 * sizeX, 2) + pow(y - 0.7 * sizeY, 2) <= pow(0.2 * sizeY, 2)) {

                        Cell2D &thisCell = cellsTemp[x * (sizeY + 1) + y];

                        thisCell.velocity.x = 0;
                        thisCell.velocity.y = 0;
                        thisCell.content = CONTENT_FLUID;
                        thisCell.index = index;
                        ++index;

                        float2 thisPos = MAC_Grid_2D::getPhysicalPos(x, y, cellPhysicalSize);
                        createParticles(particlesHost, thisPos, 1);
                    }
                }
            }

            grid.fluidCount = index;
        }

    };
}

#endif //AQUARIUS_FLUID_2D_SEMILAGRANGE_CUH
