
/*
#include "Fluid_2D_kernels.cuh"






__global__  void applyGravityImpl(Cell2D* cells, int sizeX, int sizeY, float timeStep, float gravitationalAcceleration) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= sizeX * sizeY) return;

	int x = index / sizeY;
	int y = index - x * sizeY;

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

__global__  void fixBoundaryX(Cell2D* cells, int sizeX, int sizeY) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index > sizeY) return;
	int y = index;

	get2D(cells, 0, y).newVelocity.x = 0;
	get2D(cells, sizeX, y).newVelocity.x = 0;
	get2D(cells, 0, y).hasVelocityX = true;
	get2D(cells, sizeX, y).hasVelocityX = true;
	get2D(cells, sizeX, y).content = CONTENT_SOLID;
}

__global__  void fixBoundaryY(Cell2D* cells, int sizeX, int sizeY) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index > sizeX) return;
	int x = index;

	get2D(cells, x, 0).newVelocity.y = 0;
	//get2D(cells,x,sizeY).newVelocity.y = 0;
	get2D(cells, x, 0).hasVelocityY = true;
	get2D(cells, x, sizeY).hasVelocityY = true;
	get2D(cells, x, sizeY).content = CONTENT_AIR;
}

__device__ __host__  float getNeibourCoefficient(int x, int y, float dt_div_rho_div_dx, float u, float& centerCoefficient, float& RHS, Cell2D* cells,
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
		else if (get2D(cells, x, y).content == CONTENT_AIR) {
			return 0;
		}
	}
}

__global__  void constructPressureEquations(Cell2D* cells, int sizeX, int sizeY, PressureEquation2D* equations, float dt_div_rho_div_dx, bool* hasNonZeroRHS) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= sizeX * sizeY) return;

	int x = index / sizeY;
	int y = index - x * sizeY;

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

__global__  void setPressure(Cell2D* cells, int sizeX, int sizeY, double* pressureResult) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= sizeX * sizeY) return;

	int x = index / sizeY;
	int y = index - x * sizeY;

	if (get2D(cells, x, y).content != CONTENT_FLUID)
		return;

	get2D(cells, x, y).pressure = pressureResult[get2D(cells, x, y).fluidIndex];
}

__global__  void updateVelocityWithPressureImpl(Cell2D* cells, int sizeX, int sizeY, float dt_div_rho_div_dx) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= sizeX * sizeY) return;

	int x = index / sizeY;
	int y = index - x * sizeY;

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

__global__  void extrapolateVelocityByOne(Cell2D* cells, int sizeX, int sizeY) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= sizeX * sizeY) return;

	int x = index / sizeY;
	int y = index - x * sizeY;

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

__global__  void drawCellImpl(Cell2D* cells, int sizeX, int sizeY, unsigned char* image) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= sizeX * sizeY) return;

	int x = index / sizeY;
	int y = index - x * sizeY;

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

__global__  void computeDivergenceImpl(Cell2D* cells, int sizeX, int sizeY, float cellPhysicalSize) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= sizeX * sizeY) return;

	int x = index / sizeY;
	int y = index - x * sizeY;

	Cell2D& thisCell = get2D(cells, x, y);
	Cell2D& upCell = get2D(cells, x, y + 1);
	Cell2D& rightCell = get2D(cells, x + 1, y);

	float div = (upCell.newVelocity.y - thisCell.newVelocity.y + rightCell.newVelocity.x - thisCell.newVelocity.x) / cellPhysicalSize;
	thisCell.divergence = div;

}

__global__  void resetPressureImpl(Cell2D* cells, int sizeX, int sizeY) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= sizeX * sizeY) return;

	int x = index / sizeY;
	int y = index - x * sizeY;

	Cell2D& thisCell = get2D(cells, x, y);
	thisCell.pressure = 0;

}

__global__  void jacobiImpl(Cell2D* cells, int sizeX, int sizeY, float dt_div_rho_div_dx, float cellPhysicalSize) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= sizeX * sizeY) return;

	int x = index / sizeY;
	int y = index - x * sizeY;

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

	thisCell.pressure = newPressure;
}
*/