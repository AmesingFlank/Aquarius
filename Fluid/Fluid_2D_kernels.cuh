#pragma once

#include "../GpuCommons.h"
#include "PressureEquation2D.cuh"
#include "../MAC_Grid_2D.cuh"

template<typename Particle>
__global__ inline void calcHashImpl(int* particleHashes,  // output
	Particle* particles,               // input: positions
	int particleCount,
	float cellPhysicalSize, int sizeX, int sizeY) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= particleCount) return;

	Particle& p = particles[index];

	float2 pos = p.position;

	int x = pos.x / cellPhysicalSize;
	int y = pos.y / cellPhysicalSize;
	int hash = x * (sizeY + 1) + y;

	particleHashes[index] = hash;
}



template<typename Particle>
__global__ inline void transferToParticlesImpl(Cell2D* cells, Particle* particles, int particleCount, int sizeX, int sizeY, float cellPhysicalSize) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= particleCount) return;


	Particle& particle = particles[index];
	float2 newGridVelocity = MAC_Grid_2D::getPointNewVelocity(particle.position, cellPhysicalSize, sizeX, sizeY, cells);
	float2 oldGridVelocity = MAC_Grid_2D::getPointVelocity(particle.position, cellPhysicalSize, sizeX, sizeY, cells);
	float2 velocityChange = newGridVelocity - oldGridVelocity;
	particle.velocity += velocityChange; //FLIP

	//particle.velocity = newGridVelocity; //PIC

}



template<typename Particle>
__global__ inline void moveParticlesImpl(float timeStep, Cell2D* cells, Particle* particles, int particleCount, int sizeX, int sizeY,float cellPhysicalSize) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= particleCount) return;

	Particle& particle = particles[index];
	float2 beginPos = particle.position;


	float2 u1 = MAC_Grid_2D::getPointNewVelocity(beginPos, cellPhysicalSize, sizeX, sizeY, cells);
	float2 u2 = MAC_Grid_2D::getPointNewVelocity(beginPos + timeStep * u1 / 2, cellPhysicalSize, sizeX, sizeY, cells);
	float2 u3 = MAC_Grid_2D::getPointNewVelocity(beginPos + timeStep * u2 * 3 / 4, cellPhysicalSize, sizeX, sizeY, cells);

	float2 destPos = beginPos + timeStep * (u1 * 2 / 9 + u2 * 3 / 9 + u3 * 4 / 9);

	//destPos = beginPos+particle.velocity*timeStep;


	destPos.x = max(0.0 + 1e-6, min(sizeX * cellPhysicalSize - 1e-6, destPos.x));
	destPos.y = max(0.0 + 1e-6, min(sizeY * cellPhysicalSize - 1e-6, destPos.y));

	particle.position = destPos;

}

__global__ inline void findCellStartEndImpl(int* particleHashes,
	int* cellStart, int* cellEnd,
	int particleCount) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= particleCount) return;

	int thisHash = particleHashes[index];


	if (index == 0 || particleHashes[index - 1] < thisHash) {
		cellStart[thisHash] = index;
	}

	if (index == particleCount - 1 || particleHashes[index + 1] > thisHash) {
		cellEnd[thisHash] = index;
	}
}


__global__ inline void resetAllCells(Cell2D* cells, int sizeX, int sizeY, float content) {
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

// used by FLIP
template<typename Particle>
__global__ inline void transferToCell(Cell2D* cells, int sizeX, int sizeY, float cellPhysicalSize, int* cellStart, int* cellEnd, 
	Particle* particles, int particleCount) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= sizeX * sizeY) return;

	int y = index / sizeX;
	int x = index - y * sizeX;

	int cellID = x * (sizeY + 1) + y;
	Cell2D& thisCell = get2D(cells, x, y);

	int cellsToCheck[9];
	for (int r = 0; r < 3; ++r) {
		for (int c = 0; c < 3; ++c) {
			cellsToCheck[c * 3 + r] = cellID + (r - 1) + (c - 1) * (sizeY + 1);
		}
	}

	int cellCount = (sizeX + 1) * (sizeY + 1);

	float2 xVelocityPos = make_float2(x * cellPhysicalSize, (y + 0.5) * cellPhysicalSize);
	float2 yVelocityPos = make_float2((x + 0.5) * cellPhysicalSize, y * cellPhysicalSize);

	float totalWeightX = 0;
	float totalWeightY = 0;


	for (int cell : cellsToCheck) {
		if (cell >= 0 && cell < cellCount) {
			for (int j = cellStart[cell]; j <= cellEnd[cell]; ++j) {
				if (j >= 0 && j < particleCount) {
					const Particle& p = particles[j];
					float thisWeightX = trilinearHatKernel(p.position - xVelocityPos, cellPhysicalSize);
					float thisWeightY = trilinearHatKernel(p.position - yVelocityPos, cellPhysicalSize);

					thisCell.velocity.x += thisWeightX * p.velocity.x;
					thisCell.velocity.y += thisWeightY * p.velocity.y;

					totalWeightX += thisWeightX;
					totalWeightY += thisWeightY;
				}
			}
		}
	}

	if (totalWeightX > 0)
		thisCell.velocity.x /= totalWeightX;
	if (totalWeightY > 0)
		thisCell.velocity.y /= totalWeightY;
	thisCell.newVelocity = thisCell.velocity;

	for (int j = cellStart[cellID]; j <= cellEnd[cellID]; ++j) {
		if (j >= 0 && j < particleCount) {
			thisCell.content = CONTENT_FLUID;

			const Particle& p = particles[j];


			if (p.kind > 0) {
				thisCell.fluid1Count++;
			}
			else {
				thisCell.fluid0Count++;
			}
		}
	}
}


__global__ inline void applyForcesImpl(Cell2D* cells, int sizeX, int sizeY, float timeStep, float gravitationalAcceleration) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= sizeX * sizeY) return;

	int y = index / sizeX;
	int x = index - y * sizeX;

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

__global__ inline void fixBoundaryX(Cell2D* cells, int sizeX, int sizeY) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index > sizeY) return;
	int y = index;

	get2D(cells, 0, y).newVelocity.x = 0;
	get2D(cells, sizeX, y).newVelocity.x = 0;
	get2D(cells, 0, y).hasVelocityX = true;
	get2D(cells, sizeX, y).hasVelocityX = true;
	get2D(cells, sizeX, y).content = CONTENT_SOLID;
}

__global__ inline void fixBoundaryY(Cell2D* cells, int sizeX, int sizeY) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index > sizeX) return;
	int x = index;

	get2D(cells, x, 0).newVelocity.y = 0;
	//get2D(cells,x,sizeY).newVelocity.y = 0;
	get2D(cells, x, 0).hasVelocityY = true;
	get2D(cells, x, sizeY).hasVelocityY = true;
	get2D(cells, x, sizeY).content = CONTENT_AIR;
}

__device__ __host__ inline float getNeibourCoefficient(int x, int y, float dt_div_rho_div_dx, float u, float& centerCoefficient, float& RHS, Cell2D* cells,
	int sizeX, int sizeY) {
	if (x >= 0 && x < sizeX && y >= 0 && y < sizeY && get2D(cells, x, y).content == CONTENT_FLUID) {
		return dt_div_rho_div_dx * -1;
	}
	else {
		if (x < 0 || y < 0 || x >= sizeX || get2D(cells, x, y).content == CONTENT_SOLID) {
			centerCoefficient -= dt_div_rho_div_dx;
			//RHS += u;
			return 0;
		}
		else if (y >= sizeY || get2D(cells, x, y).content == CONTENT_AIR) {
			return 0;
		}
	}
}

__global__ inline void constructPressureEquations(Cell2D* cells, int sizeX, int sizeY, PressureEquation2D* equations, float dt_div_rho_div_dx,bool* hasNonZeroRHS) {

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

__global__ inline void setPressure(Cell2D* cells, int sizeX, int sizeY, double* pressureResult) {
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

__global__ inline void extrapolateVelocityByOne(Cell2D* cells, int sizeX, int sizeY) {
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

__global__ inline void drawCellImpl(Cell2D* cells, int sizeX, int sizeY, unsigned char* image) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= sizeX * sizeY) return;

	int y = index / sizeX;
	int x = index - y * sizeX;

	Cell2D& thisCell = get2D(cells, x, y);
	unsigned char* base = image + 4 * (sizeX * y + x);

	int cellID = x * (sizeY + 1) + y;

	if (thisCell.content == CONTENT_FLUID) {
		float fluid1percentage = thisCell.fluid1Count / (thisCell.fluid1Count + thisCell.fluid0Count);
		base[0] = 255 * fluid1percentage;
		base[1] = 0;
		base[2] = 255 * (1 - fluid1percentage);

		thisCell.fluid1Count = thisCell.fluid0Count = 0;
	}
	else {

		base[0] = 255;
		base[1] = 255;
		base[2] = 255;

	}
	base[3] = 255;

}

template<typename Particle>
__global__ inline void drawParticleImpl(int sizeX, int sizeY, float cellPhysicalSize, Particle* particles, int particleCount,
	unsigned char* image, int imageSizeX, int imageSizeY) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= particleCount) return;

	Particle& particle = particles[index];

	float gridSizeX = sizeX * cellPhysicalSize;
	float gridSizeY = sizeY * cellPhysicalSize;

	int x = (float)imageSizeX * particle.position.x / gridSizeX;
	int y = (float)imageSizeY * particle.position.y / gridSizeY;
	unsigned char* base = image + (y * imageSizeX + x) * 4;

	if (particle.kind == 0) {
		base[0] = 0;
		base[1] = 0;
		base[2] = 255;
	}
	else {
		base[0] = 255;
		base[1] = 0;
		base[2] = 0;
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