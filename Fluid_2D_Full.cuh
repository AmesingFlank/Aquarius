//
// Created by AmesingFlank on 2019-04-16.
//

#ifndef AQUARIUS_FLUID_2D_Full_CUH
#define AQUARIUS_FLUID_2D_Full_CUH

#include "MAC_Grid_2D.cuh"
#include "SPD_Solver.h"
#include <vector>
#include <utility>
#include "GpuCommons.h"
#include "Fluid_2D.h"
#include <unordered_map>
#include <thrust/functional.h>
#include <thrust/reduce.h>



namespace Fluid_2D_Full {

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

	

	__global__
		inline
		void resetAllCells(Cell2D* cells, int sizeX, int sizeY, float content) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= sizeX * sizeY) return;

		int y = index / sizeX;
		int x = index - y * sizeX;

		Cell2D& thisCell = get2D(cells, x, y);

		thisCell.content = content;
		thisCell.velocity = make_float2(0, 0);
		thisCell.newVelocity = make_float2(0, 0);
		thisCell.fluid0Count = 0;
		thisCell.fluid1Count = 0;

	}

	__global__
		inline
		void applyForcesImpl(Cell2D* cells, int sizeX, int sizeY, float timeStep, float gravitationalAcceleration) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= sizeX * sizeY) return;

		int y = index / sizeX;
		int x = index - y * sizeX;

		float xRel = (float)x / (float)sizeX;
		float yRel = (float)y / (float)sizeY;
		if (xRel >= 0.49 && xRel <= 0.51 &&  yRel >= 0.4 && yRel <= 0.6) {
			get2D(cells, x, y).newVelocity.y -= gravitationalAcceleration * timeStep;
		}
		return;

		if (get2D(cells, x, y).content == CONTENT_FLUID) {
			get2D(cells, x, y).newVelocity.y -= gravitationalAcceleration * timeStep;
			if (get2D(cells, x, y + 1).content == CONTENT_AIR)
				get2D(cells, x, y + 1).newVelocity.y -= gravitationalAcceleration * timeStep;
		}
		else if (get2D(cells, x, y).content == CONTENT_AIR) {
			//if( x-1 >0 && grid.get2D(cells,x-1,y).content == CONTENT_AIR) grid.get2D(cells,x,y).newVelocity.x = 0;
			//if( y-1 >0 && grid.get2D(cells,x,y-1).content == CONTENT_AIR) grid.get2D(cells,x,y).newVelocity.y = 0;
		}

	}

	__global__
		inline
		void fixBoundaryX(Cell2D* cells, int sizeX, int sizeY) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index > sizeY) return;
		int y = index;

		get2D(cells, 0, y).newVelocity.x = 0;
		get2D(cells, sizeX, y).newVelocity.x = 0;
		get2D(cells, 0, y).hasVelocityX = true;
		get2D(cells, sizeX, y).hasVelocityX = true;
		get2D(cells, sizeX, y).content = CONTENT_SOLID;
	}

	__global__
		inline
		void fixBoundaryY(Cell2D* cells, int sizeX, int sizeY) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index > sizeX) return;
		int x = index;

		get2D(cells, x, 0).newVelocity.y = 0;
		get2D(cells,x,sizeY).newVelocity.y = 0;
		get2D(cells, x, 0).hasVelocityY = true;
		get2D(cells, x, sizeY).hasVelocityY = true;
		get2D(cells, x, sizeY).content = CONTENT_SOLID;
	}

	__device__ __host__ inline float getNeibourCoefficient(int x, int y, float dt_div_rho_div_dx, float u, float& centerCoefficient, float& RHS, Cell2D* cells,
			int sizeX, int sizeY) {
		if (x >= 0 && x < sizeX && y >= 0 && y < sizeY && get2D(cells, x, y).content == CONTENT_FLUID) {
			return dt_div_rho_div_dx * -1;
		}
		else {
			if (x < 0 || y < 0 || x >= sizeX || y >= sizeY || get2D(cells, x, y).content == CONTENT_SOLID) {
				centerCoefficient -= dt_div_rho_div_dx;
				//RHS += u;
				return 0;
			}
			else if ( get2D(cells, x, y).content == CONTENT_AIR) {
				return 0;
			}
		}
	}

	__global__
		inline
		void constructPressureEquations(Cell2D* cells, int sizeX, int sizeY, PressureEquation2D* equations, float dt_div_rho_div_dx,
			bool* hasNonZeroRHS) {

		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= sizeX * sizeY) return;

		int y = index / sizeX;
		int x = index - y * sizeX;

		get2D(cells, x, y).pressure = 0;
		if (get2D(cells, x, y).content != CONTENT_FLUID)
			return;
		Cell2D& thisCell = get2D(cells, x, y);
		Cell2D& rightCell = get2D(cells, x + 1, y);
		Cell2D& upCell = get2D(cells, x, y + 1);

		PressureEquation2D thisEquation;
		float RHS = (thisCell.newVelocity.y - upCell.newVelocity.y + thisCell.newVelocity.x - rightCell.newVelocity.x);

		float centerCoeff = dt_div_rho_div_dx * 4;

		float leftCoeff = getNeibourCoefficient(x - 1, y, dt_div_rho_div_dx, thisCell.newVelocity.x, centerCoeff, RHS, cells, sizeX, sizeY);
		float rightCoeff = getNeibourCoefficient(x + 1, y, dt_div_rho_div_dx, rightCell.newVelocity.x, centerCoeff, RHS, cells, sizeX, sizeY);
		float downCoeff = getNeibourCoefficient(x, y - 1, dt_div_rho_div_dx, thisCell.newVelocity.y, centerCoeff, RHS, cells, sizeX, sizeY);
		float upCoeff = getNeibourCoefficient(x, y + 1, dt_div_rho_div_dx, upCell.newVelocity.y, centerCoeff, RHS, cells, sizeX, sizeY);

		int nnz = 0;

		if (downCoeff) {
			Cell2D& downCell = get2D(cells, x, y - 1);
			thisEquation.termsIndex[thisEquation.termCount] = downCell.fluidIndex;
			thisEquation.termsCoeff[thisEquation.termCount] = downCoeff;
			++thisEquation.termCount;
			++nnz;
		}
		if (leftCoeff) {
			Cell2D& leftCell = get2D(cells, x - 1, y);
			thisEquation.termsIndex[thisEquation.termCount] = leftCell.fluidIndex;
			thisEquation.termsCoeff[thisEquation.termCount] = leftCoeff;
			++thisEquation.termCount;
			++nnz;
		}
		thisEquation.termsIndex[thisEquation.termCount] = thisCell.fluidIndex;
		thisEquation.termsCoeff[thisEquation.termCount] = centerCoeff;
		++thisEquation.termCount;
		if (rightCoeff) {
			thisEquation.termsIndex[thisEquation.termCount] = rightCell.fluidIndex;
			thisEquation.termsCoeff[thisEquation.termCount] = rightCoeff;
			++thisEquation.termCount;
			++nnz;
		}
		if (upCoeff) {
			thisEquation.termsIndex[thisEquation.termCount] = upCell.fluidIndex;
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
		equations[thisCell.fluidIndex] = thisEquation;

	}

	__global__
		inline
		void setPressure(Cell2D* cells, int sizeX, int sizeY, double* pressureResult) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= sizeX * sizeY) return;

		int y = index / sizeX;
		int x = index - y * sizeX;

		if (get2D(cells, x, y).content != CONTENT_FLUID)
			return;

		get2D(cells, x, y).pressure = pressureResult[get2D(cells, x, y).fluidIndex];
	}

	__global__ inline void updateVelocityWithPressureImpl(Cell2D* cells, int sizeX, int sizeY, float dt_div_rho_div_dx) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= sizeX * sizeY) return;

		int y = index / sizeX;
		int x = index - y * sizeX;

		Cell2D& thisCell = get2D(cells, x, y);

		thisCell.hasVelocityX = false;
		thisCell.hasVelocityY = false;

		if (x > 0) {
			Cell2D& leftCell = get2D(cells, x - 1, y);
			if (thisCell.content == CONTENT_FLUID || leftCell.content == CONTENT_FLUID) {
				float uX = thisCell.newVelocity.x - dt_div_rho_div_dx * (thisCell.pressure - leftCell.pressure);
				thisCell.newVelocity.x = uX;
				thisCell.hasVelocityX = true;
			}
		}
		if (y > 0) {
			Cell2D& downCell = get2D(cells, x, y - 1);
			if (thisCell.content == CONTENT_FLUID || downCell.content == CONTENT_FLUID) {
				float uY = thisCell.newVelocity.y - dt_div_rho_div_dx * (thisCell.pressure - downCell.pressure);
				thisCell.newVelocity.y = uY;
				thisCell.hasVelocityY = true;
			}
		}
	}

	__global__
		inline
		void extrapolateVelocityByOne(Cell2D* cells, int sizeX, int sizeY) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= sizeX * sizeY) return;

		int y = index / sizeX;
		int x = index - y * sizeX;

		Cell2D& thisCell = get2D(cells, x, y);
		const float epsilon = 1e-6;
		if (!thisCell.hasVelocityX) {
			float sumNeighborX = 0;
			int neighborXCount = 0;
			if (x > 0) {
				Cell2D& leftCell = get2D(cells, x - 1, y);
				if (leftCell.hasVelocityX && leftCell.newVelocity.x > epsilon) {
					sumNeighborX += leftCell.newVelocity.x;
					neighborXCount++;
				}
			}
			if (y > 0) {
				Cell2D& downCell = get2D(cells, x, y - 1);
				if (downCell.hasVelocityX && downCell.newVelocity.y > epsilon) {
					sumNeighborX += downCell.newVelocity.x;
					neighborXCount++;
				}
			}
			if (x < sizeX - 1) {
				Cell2D& rightCell = get2D(cells, x + 1, y);
				if (rightCell.hasVelocityX && rightCell.newVelocity.x < -epsilon) {
					sumNeighborX += rightCell.newVelocity.x;
					neighborXCount++;
				}
			}
			if (y < sizeY - 1) {
				Cell2D& upCell = get2D(cells, x, y + 1);
				if (upCell.hasVelocityX && upCell.newVelocity.y < -epsilon) {
					sumNeighborX += upCell.newVelocity.x;
					neighborXCount++;
				}
			}
			if (neighborXCount > 0) {
				thisCell.newVelocity.x = sumNeighborX / (float)neighborXCount;
				thisCell.hasVelocityX = true;
			}
		}

		if (!thisCell.hasVelocityY) {
			float sumNeighborY = 0;
			int neighborYCount = 0;
			if (x > 0) {
				Cell2D& leftCell = get2D(cells, x - 1, y);
				if (leftCell.hasVelocityY && leftCell.newVelocity.x > epsilon) {
					sumNeighborY += leftCell.newVelocity.y;
					neighborYCount++;
				}
			}
			if (y > 0) {
				Cell2D& downCell = get2D(cells, x, y - 1);
				if (downCell.hasVelocityY && downCell.newVelocity.y > epsilon) {
					sumNeighborY += downCell.newVelocity.y;
					neighborYCount++;
				}
			}
			if (x < sizeX - 1) {
				Cell2D& rightCell = get2D(cells, x + 1, y);
				if (rightCell.hasVelocityY && rightCell.newVelocity.x < -epsilon) {
					sumNeighborY += rightCell.newVelocity.y;
					neighborYCount++;
				}
			}
			if (y < sizeY - 1) {
				Cell2D& upCell = get2D(cells, x, y + 1);
				if (upCell.hasVelocityY && upCell.newVelocity.y < -epsilon) {
					sumNeighborY += upCell.newVelocity.y;
					neighborYCount++;
				}
			}
			if (neighborYCount > 0) {
				thisCell.newVelocity.y = sumNeighborY / (float)neighborYCount;
				thisCell.hasVelocityY = true;
			}
		}
	}

	__global__
		inline
		void updateTextureImpl(Cell2D* cells, int sizeX, int sizeY, unsigned char* image,float maxSpeed) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= sizeX * sizeY) return;

		int y = index / sizeX;
		int x = index - y * sizeX;

		Cell2D& thisCell = get2D(cells, x, y);
		unsigned char* base = image + 4 * (sizeX * y + x);

		int cellID = x * (sizeY + 1) + y;

		if (thisCell.content == CONTENT_FLUID) {
			
			base[0] = 255 * (thisCell.divergence / length(thisCell.velocity)) ;
			base[1] = 0;
			base[2] = 0;

			thisCell.fluid1Count = thisCell.fluid0Count = 0;
		}
		else {
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


	__global__ inline void computeDivergenceImpl(Cell2D* cells, int sizeX, int sizeY, float cellPhysicalSize) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= sizeX * sizeY) return;

		int y = index / sizeX;
		int x = index - y * sizeX;

		Cell2D& thisCell = get2D(cells, x, y);
		Cell2D& upCell = get2D(cells, x, y + 1);
		Cell2D& rightCell = get2D(cells, x + 1, y);

		float div = (upCell.newVelocity.y - thisCell.newVelocity.y + rightCell.newVelocity.x - thisCell.newVelocity.x) / cellPhysicalSize;
		thisCell.divergence = div;

	}

	__global__ inline void resetPressureImpl(Cell2D* cells, int sizeX, int sizeY) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= sizeX * sizeY) return;

		int y = index / sizeX;
		int x = index - y * sizeX;

		Cell2D& thisCell = get2D(cells, x, y);
		thisCell.pressure = 0;

	}

	__global__ inline void jacobiImpl(Cell2D* cells, int sizeX, int sizeY, float dt_div_rho_div_dx, float cellPhysicalSize) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= sizeX * sizeY) return;

		int y = index / sizeX;
		int x = index - y * sizeX;

		Cell2D& thisCell = get2D(cells, x, y);
		Cell2D& rightCell = get2D(cells, x + 1, y);
		Cell2D& upCell = get2D(cells, x, y + 1);

		if (thisCell.content == CONTENT_AIR) {
			thisCell.pressure = 0;
			return;
		}

		float RHS = -thisCell.divergence * cellPhysicalSize;

		float newPressure = 0;

		float centerCoeff = dt_div_rho_div_dx * 4;

		if (x > 0)
			newPressure += get2D(cells, x - 1, y).pressure * dt_div_rho_div_dx;
		else {
			centerCoeff -= dt_div_rho_div_dx;
		}

		if (x < sizeX - 1)
			newPressure += get2D(cells, x + 1, y).pressure * dt_div_rho_div_dx;
		else {
			centerCoeff -= dt_div_rho_div_dx;
		}

		if (y > 0)
			newPressure += get2D(cells, x, y - 1).pressure * dt_div_rho_div_dx;
		else {
			centerCoeff -= dt_div_rho_div_dx;
		}

		if (y < sizeY - 1)
			newPressure += get2D(cells, x, y + 1).pressure * dt_div_rho_div_dx;
		else {
			centerCoeff -= dt_div_rho_div_dx;
		}


		newPressure += RHS;
		newPressure /= centerCoeff;


		/*if (y==0 && x==100) {

			printf("np: %f \n", newPressure);
			printf("yup: %f \n", upCell.newVelocity.y);
			printf("ythis: %f \n", thisCell.newVelocity.y);

			printf("div: %f \n\n", thisCell.divergence);

		}*/



		thisCell.pressure = newPressure;
	}


	__global__ inline void advectVelocityImpl(Cell2D* cells, int sizeX, int sizeY, float timeStep, float gravitationalAcceleration,
			float cellPhysicalSize) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= (sizeX + 1) * (sizeY + 1)) return;

		int x = index / (sizeY + 1);
		int y = index - x * (sizeY + 1);

		if (get2D(cells, x, y).content == CONTENT_AIR) return;
		float2 thisVelocity = MAC_Grid_2D::getCellVelocity(x, y, sizeX, sizeY, cells);
		float2 thisPos = MAC_Grid_2D::getPhysicalPos(x, y, cellPhysicalSize);

		float2 u1 = MAC_Grid_2D::getPointVelocity(thisPos, cellPhysicalSize, sizeX, sizeY, cells);
		float2 u2 = MAC_Grid_2D::getPointVelocity(thisPos - timeStep * u1 / 2, cellPhysicalSize, sizeX, sizeY, cells);
		float2 u3 = MAC_Grid_2D::getPointVelocity(thisPos - timeStep * u2 * 3 / 4, cellPhysicalSize, sizeX, sizeY, cells);

		float2 sourcePos = thisPos - timeStep * (u1 * 2 / 9 + u2 * 3 / 9 + u3 * 4 / 9);


		float2 sourceVelocity =
			MAC_Grid_2D::getPointVelocity(sourcePos, cellPhysicalSize, sizeX, sizeY, cells);
		get2D(cells, x, y).newVelocity = sourceVelocity;
		if (y + 1 <= sizeY && get2D(cells, x, y + 1).content == CONTENT_AIR) {
			get2D(cells, x, y + 1).newVelocity.y = sourceVelocity.y;
		}
		if (x + 1 <= sizeX && get2D(cells, x + 1, y).content == CONTENT_AIR) {
			get2D(cells, x + 1, y).newVelocity.x = sourceVelocity.x;
		}
	}

	__global__ inline void commitVelocityChanges(Cell2D* cells, int sizeX, int sizeY) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= sizeX * sizeY) return;

		int y = index / sizeX;
		int x = index - y * sizeX;

		Cell2D& thisCell = get2D(cells, x, y);
		thisCell.velocity = thisCell.newVelocity;
	}



	class Fluid : public Fluid_2D {
	public:
		const int sizeX = 256;
		const int sizeY = 128;
		const int cellCount = (sizeX + 1) * (sizeY + 1);


		const float cellPhysicalSize = 10.f / (float)sizeY;
		const float gravitationalAcceleration = 9.8;
		const float density = 1;
		MAC_Grid_2D grid = MAC_Grid_2D(sizeX, sizeY, cellPhysicalSize);


		int numThreadsCell, numBlocksCell;


		Fluid() {
			init();
		}

		void init() {


			//set everything to air first

			Cell2D* cellsTemp = grid.copyCellsToHost();


			grid.fluidCount = 0;
			createFluid( cellsTemp);

			grid.copyCellsToDevice(cellsTemp);
			delete[]cellsTemp;


			numThreadsCell = min(1024, sizeX * sizeY);
			numBlocksCell = divUp(sizeX * sizeY, numThreadsCell);

			std::cout << numThreadsCell << std::endl << numBlocksCell << std::endl;

			fixBoundary();

		}


		void simulationStep(float totalTime) {
			float thisTimeStep = 0.05f;


			advectVelocity(thisTimeStep);

			applyForces(thisTimeStep);

			fixBoundary();

			solvePressure(thisTimeStep);

			//solvePressureJacobi(thisTimeStep);


			updateVelocityWithPressure(thisTimeStep);

			commitVelocityChanges << < numBlocksCell, numThreadsCell >> > (grid.cells, sizeX, sizeY);
			CHECK_CUDA_ERROR("commit velocity");

		}

		void advectVelocity(float timeStep) {
			advectVelocityImpl << < numBlocksCell, numThreadsCell >> >
				(grid.cells, sizeX, sizeY, timeStep, gravitationalAcceleration, cellPhysicalSize);
			cudaDeviceSynchronize();
			CHECK_CUDA_ERROR("advect velocity");

		}



		void applyForces(float timeStep) {
			applyForcesImpl << < numBlocksCell, numThreadsCell >> >
				(grid.cells, sizeX, sizeY, timeStep, gravitationalAcceleration);
			cudaDeviceSynchronize();
			CHECK_CUDA_ERROR("apply forces");

		}

		void fixBoundary() {
			int numThreads, numBlocks;

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

		void solvePressureJacobi(float timeStep) {
			computeDivergenceImpl << < numBlocksCell, numThreadsCell >> > (grid.cells, sizeX, sizeY, cellPhysicalSize);
			resetPressureImpl << < numBlocksCell, numThreadsCell >> > (grid.cells, sizeX, sizeY);


			float dt_div_rho_div_dx = timeStep / (density * cellPhysicalSize);

			for (int i = 0; i < 20; ++i) {
				jacobiImpl << < numBlocksCell, numThreadsCell >> > (grid.cells, sizeX, sizeY, dt_div_rho_div_dx, cellPhysicalSize);
			}

		}

		void solvePressure(float timeStep) {


			PressureEquation2D* equations = new PressureEquation2D[grid.fluidCount];
			int nnz = 0;
			bool hasNonZeroRHS = false;
			float dt_div_rho_div_dx = timeStep / (density * cellPhysicalSize);


			PressureEquation2D* equationsDevice;
			HANDLE_ERROR(cudaMalloc(&equationsDevice, grid.fluidCount * sizeof(PressureEquation2D)));

			bool* hasNonZeroRHS_Device;
			HANDLE_ERROR(cudaMalloc(&hasNonZeroRHS_Device, sizeof(*hasNonZeroRHS_Device)));
			HANDLE_ERROR(cudaMemset(hasNonZeroRHS_Device, 0, sizeof(*hasNonZeroRHS_Device)));

			constructPressureEquations << < numBlocksCell, numThreadsCell >> >
				(grid.cells, sizeX, sizeY, equationsDevice, dt_div_rho_div_dx, hasNonZeroRHS_Device);
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

			//std::cout<<"nnz is "<<nnz<<std::endl;


			if (!hasNonZeroRHS) {
				std::cout << "zero RHS" << std::endl;
				return;
			}


			//number of rows == number of variables == number of fluid cells
			int numVariables = grid.fluidCount;



			//construct the matrix of the linear equations
			int nnz_A = nnz;
			double* A_host = (double*)malloc(nnz_A * sizeof(*A_host));
			int* A_rowPtr_host = (int*)malloc((numVariables + 1) * sizeof(*A_rowPtr_host));
			int* A_colInd_host = (int*)malloc(nnz_A * sizeof(*A_colInd_host));

			//construct a symmetric copy, used for computing the preconditioner
			int nnz_R = (nnz - numVariables) / 2 + numVariables;
			nnz_R = numVariables;
			double* R_host = (double*)malloc(nnz_R * sizeof(*R_host));
			int* R_rowPtr_host = (int*)malloc((numVariables + 1) * sizeof(*R_rowPtr_host));
			int* R_colInd_host = (int*)malloc(nnz_R * sizeof(*R_colInd_host));

			for (int row = 0, i = 0; row < numVariables; ++row) {
				PressureEquation2D& thisEquation = equations[row];
				A_rowPtr_host[row] = i;

				for (int term = 0; term < thisEquation.termCount; ++term) {
					//if(thisEquation.termsIndex[term] > row) continue;
					A_host[i] = thisEquation.termsCoeff[term];
					A_colInd_host[i] = thisEquation.termsIndex[term];
					++i;
				}

			}

			for (int row = 0, i = 0; row < numVariables; ++row) {
				PressureEquation2D& thisEquation = equations[row];
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

			double* A_device;
			HANDLE_ERROR(cudaMalloc(&A_device, nnz_A * sizeof(*A_device)));
			HANDLE_ERROR(cudaMemcpy(A_device, A_host, nnz_A * sizeof(*A_device), cudaMemcpyHostToDevice));

			int* A_rowPtr_device;
			HANDLE_ERROR(cudaMalloc(&A_rowPtr_device, (numVariables + 1) * sizeof(*A_rowPtr_device)));
			HANDLE_ERROR(cudaMemcpy(A_rowPtr_device, A_rowPtr_host, (numVariables + 1) * sizeof(*A_rowPtr_device),
				cudaMemcpyHostToDevice));

			int* A_colInd_device;
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

			double* R_device;
			HANDLE_ERROR(cudaMalloc(&R_device, nnz_R * sizeof(*R_device)));
			HANDLE_ERROR(cudaMemcpy(R_device, R_host, nnz_R * sizeof(*R_device), cudaMemcpyHostToDevice));

			int* R_rowPtr_device;
			HANDLE_ERROR(cudaMalloc(&R_rowPtr_device, (numVariables + 1) * sizeof(*R_rowPtr_device)));
			HANDLE_ERROR(cudaMemcpy(R_rowPtr_device, R_rowPtr_host, (numVariables + 1) * sizeof(*R_rowPtr_device),
				cudaMemcpyHostToDevice));

			int* R_colInd_device;
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
			double* f_host = (double*)malloc(numVariables * sizeof(*f_host));
			for (int i = 0; i < numVariables; ++i) {
				f_host[i] = equations[i].RHS;
			}

			//solve the pressure equation
			double* result_device = solveSPD4(A, R, f_host, numVariables);

			double* result_host = new double[numVariables];
			HANDLE_ERROR(cudaMemcpy(result_host, result_device, numVariables * sizeof(*result_host),
				cudaMemcpyDeviceToHost));


			setPressure << < numBlocksCell, numThreadsCell >> > (grid.cells, sizeX, sizeY, result_device);
			cudaDeviceSynchronize();
			CHECK_CUDA_ERROR("set pressure");



			A.free();
			R.free();
			free(f_host);
			HANDLE_ERROR(cudaFree(result_device));
			delete[](result_host);

			delete[] equations;

		}

		void updateVelocityWithPressure(float timeStep) {
			float dt_div_rho_div_dx = timeStep / (density * cellPhysicalSize);
			updateVelocityWithPressureImpl << < numBlocksCell, numThreadsCell >> > (grid.cells, sizeX, sizeY, dt_div_rho_div_dx);
			cudaDeviceSynchronize();
			CHECK_CUDA_ERROR("update velocity with pressure");
		}


		void extrapolateVelocity(float timeStep) {

			//used to decide how far to extrapolate
			float maxSpeed = grid.getMaxSpeed();

			float maxDist = (maxSpeed * timeStep + 1) / cellPhysicalSize;
			//maxDist=4;
			//std::cout<<"maxDist "<<maxDist<<std::endl;

			for (int distance = 0; distance < maxDist; ++distance) {
				extrapolateVelocityByOne << < numBlocksCell, numThreadsCell >> > (grid.cells, sizeX, sizeY);
				cudaDeviceSynchronize();
				CHECK_CUDA_ERROR("extrapolate vel");
			}
		}


		virtual void updateTexture() override {
			computeDivergenceImpl << < numBlocksCell, numThreadsCell >> > (grid.cells, sizeX, sizeY, cellPhysicalSize);

			printGLError();
			glBindTexture(GL_TEXTURE_2D, texture);
			int imageMemorySize = sizeX * sizeY * 4;
			unsigned char* image = (unsigned char*)malloc(imageMemorySize);
			unsigned char* imageDevice;
			HANDLE_ERROR(cudaMalloc(&imageDevice, imageMemorySize));

			float maxSpeed = grid.getMaxSpeed();
			if (maxSpeed == 0) {
				maxSpeed = 1;
			}

			updateTextureImpl << < numBlocksCell, numThreadsCell >> > (grid.cells, sizeX, sizeY, imageDevice,maxSpeed);
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


		void createFluid( Cell2D* cellsTemp, int startIndex = 0) {
			int index = startIndex;
			for (int y = 0 * sizeY; y < sizeY; ++y) {
				for (int x = 0 * sizeX; x < 1 * sizeX; ++x) {

					Cell2D& thisCell = cellsTemp[x * (sizeY + 1) + y];

					thisCell.velocity.x = 0;
					thisCell.velocity.y = 0;
					thisCell.newVelocity = make_float2(0, 0);
					thisCell.content = CONTENT_FLUID;
					thisCell.fluidIndex = index;
					++index;
					float2 thisPos = MAC_Grid_2D::getPhysicalPos(x, y, cellPhysicalSize);
				}
			}

			grid.fluidCount = index;
		}

	};
}

#endif //AQUARIUS_FLUID_2D_Full_CUH
