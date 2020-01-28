#include "Fluid_3D_kernels.cuh"



__global__  void applyGravityImpl(Cell3D* cells, int sizeX, int sizeY, int sizeZ, float timeStep, float gravitationalAcceleration) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= sizeX * sizeY * sizeZ) return;

	int x = index / (sizeY * sizeZ);
	int y = (index - x * (sizeY * sizeZ)) / sizeZ;
	int z = index - x * (sizeY * sizeZ) - y * (sizeZ);

	if (get3D(cells, x, y, z).content == CONTENT_FLUID) {
		get3D(cells, x, y, z).newVelocity.y -= gravitationalAcceleration * timeStep;
		if (get3D(cells, x, y + 1, z).content == CONTENT_AIR) {
			get3D(cells, x, y + 1, z).newVelocity.y -= gravitationalAcceleration * timeStep;
		}
	}
}




__global__  void fixBoundaryX(Cell3D* cells, int sizeX, int sizeY, int sizeZ) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= (sizeY + 1) * (sizeZ + 1)) return;

	int y = index / (sizeZ + 1);
	int z = index - y * (sizeZ + 1);

	get3D(cells, 0, y, z).newVelocity.x = 0;
	get3D(cells, sizeX, y, z).newVelocity.x = 0;
	get3D(cells, 0, y, z).hasVelocityX = true;
	get3D(cells, sizeX, y, z).hasVelocityX = true;

	get3D(cells, sizeX, y, z).content = CONTENT_SOLID;
}

__global__  void fixBoundaryY(Cell3D* cells, int sizeX, int sizeY, int sizeZ) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= (sizeX + 1) * (sizeZ + 1)) return;

	int x = index / (sizeZ + 1);
	int z = index - x * (sizeZ + 1);

	get3D(cells, x, 0, z).newVelocity.y = 0;
	get3D(cells, x, sizeY, z).newVelocity.y = 0;
	get3D(cells, x, 0, z).hasVelocityY = true;
	get3D(cells, x, sizeY, z).hasVelocityY = true;
	get3D(cells, x, sizeY, z).content = CONTENT_SOLID;
}

__global__  void fixBoundaryZ(Cell3D* cells, int sizeX, int sizeY, int sizeZ) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= (sizeX + 1) * (sizeY + 1)) return;

	int x = index / (sizeY + 1);
	int y = index - x * (sizeY + 1);

	get3D(cells, x, y, 0).newVelocity.z = 0;
	get3D(cells, x, y, sizeZ).newVelocity.z = 0;
	get3D(cells, x, y, 0).hasVelocityZ = true;
	get3D(cells, x, y, sizeZ).hasVelocityZ = true;

	get3D(cells, x, y, sizeZ).content = CONTENT_SOLID;
}


__device__ __host__  float getNeibourCoefficient(int x, int y, int z, float u, float& centerCoefficient, float& RHS, Cell3D* cells,
	int sizeX, int sizeY, int sizeZ) {
	if (x >= 0 && x < sizeX && y >= 0 && y < sizeY && z >= 0 && z < sizeZ && get3D(cells, x, y, z).content == CONTENT_FLUID) {
		return -1;
	}
	else {
		if (x < 0 || y < 0 || z < 0 || x >= sizeX || y >= sizeY || z >= sizeZ || get3D(cells, x, y, z).content == CONTENT_SOLID) {
			centerCoefficient -= 1;
			//RHS += u;
			return 0;
		}
		else if (get3D(cells, x, y, z).content == CONTENT_AIR) {
			return 0;
		}
	}
}



__global__  void constructPressureEquations(Cell3D* cells, int sizeX, int sizeY, int sizeZ, PressureEquation3D* equations, bool* hasNonZeroRHS) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= sizeX * sizeY * sizeZ) return;

	int x = index / (sizeY * sizeZ);
	int y = (index - x * (sizeY * sizeZ)) / sizeZ;
	int z = index - x * (sizeY * sizeZ) - y * (sizeZ);

	get3D(cells, x, y, z).pressure = 0;
	if (get3D(cells, x, y, z).content != CONTENT_FLUID)
		return;

	Cell3D& thisCell = get3D(cells, x, y, z);
	Cell3D& rightCell = get3D(cells, x + 1, y, z);
	Cell3D& upCell = get3D(cells, x, y + 1, z);
	Cell3D& frontCell = get3D(cells, x, y, z + 1);


	PressureEquation3D thisEquation;
	float RHS = -thisCell.divergence;

	float centerCoeff = 6;

	float leftCoeff = getNeibourCoefficient(x - 1, y, z, thisCell.newVelocity.x, centerCoeff, RHS, cells, sizeX, sizeY, sizeZ);
	float rightCoeff = getNeibourCoefficient(x + 1, y, z, rightCell.newVelocity.x, centerCoeff, RHS, cells, sizeX, sizeY, sizeZ);
	float downCoeff = getNeibourCoefficient(x, y - 1, z, thisCell.newVelocity.y, centerCoeff, RHS, cells, sizeX, sizeY, sizeZ);
	float upCoeff = getNeibourCoefficient(x, y + 1, z,  upCell.newVelocity.y, centerCoeff, RHS, cells, sizeX, sizeY, sizeZ);
	float backCoeff = getNeibourCoefficient(x, y, z - 1, thisCell.newVelocity.z, centerCoeff, RHS, cells, sizeX, sizeY, sizeZ);
	float frontCoeff = getNeibourCoefficient(x, y, z + 1,  frontCell.newVelocity.z, centerCoeff, RHS, cells, sizeX, sizeY, sizeZ);

	int nnz = 0;

	if (downCoeff) {
		Cell3D& downCell = get3D(cells, x, y - 1, z);
		thisEquation.termsIndex[thisEquation.termCount] = downCell.fluidIndex;
		thisEquation.termsCoeff[thisEquation.termCount] = downCoeff;
		++thisEquation.termCount;
		++nnz;
	}
	if (leftCoeff) {
		Cell3D& leftCell = get3D(cells, x - 1, y, z);
		thisEquation.termsIndex[thisEquation.termCount] = leftCell.fluidIndex;
		thisEquation.termsCoeff[thisEquation.termCount] = leftCoeff;
		++thisEquation.termCount;
		++nnz;
	}
	if (backCoeff) {
		Cell3D& backCell = get3D(cells, x, y, z - 1);
		thisEquation.termsIndex[thisEquation.termCount] = backCell.fluidIndex;
		thisEquation.termsCoeff[thisEquation.termCount] = backCoeff;
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
	if (frontCoeff) {
		thisEquation.termsIndex[thisEquation.termCount] = frontCell.fluidIndex;
		thisEquation.termsCoeff[thisEquation.termCount] = frontCoeff;
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
	thisEquation.z = z;
	equations[thisCell.fluidIndex] = thisEquation;

}

__global__  void setPressure(Cell3D* cells, int sizeX, int sizeY, int sizeZ, double* pressureResult) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= sizeX * sizeY * sizeZ) return;

	int x = index / (sizeY * sizeZ);
	int y = (index - x * (sizeY * sizeZ)) / sizeZ;
	int z = index - x * (sizeY * sizeZ) - y * (sizeZ);

	if (get3D(cells, x, y, z).content != CONTENT_FLUID)
		return;

	get3D(cells, x, y, z).pressure = pressureResult[get3D(cells, x, y, z).fluidIndex];
}


__global__  void updateVelocityWithPressureImpl(Cell3D* cells, int sizeX, int sizeY, int sizeZ) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= sizeX * sizeY * sizeZ) return;

	int x = index / (sizeY * sizeZ);
	int y = (index - x * (sizeY * sizeZ)) / sizeZ;
	int z = index - x * (sizeY * sizeZ) - y * (sizeZ);

	Cell3D& thisCell = get3D(cells, x, y, z);

	thisCell.hasVelocityX = false;
	thisCell.hasVelocityY = false;
	thisCell.hasVelocityZ = false;

	if (x > 0) {
		Cell3D& leftCell = get3D(cells, x - 1, y, z);
		if (thisCell.content == CONTENT_FLUID || leftCell.content == CONTENT_FLUID) {
			float uX = thisCell.newVelocity.x -  (thisCell.pressure - leftCell.pressure);
			thisCell.newVelocity.x = uX;
			thisCell.hasVelocityX = true;
		}
	}
	if (y > 0) {
		Cell3D& downCell = get3D(cells, x, y - 1, z);
		if (thisCell.content == CONTENT_FLUID || downCell.content == CONTENT_FLUID) {
			float uY = thisCell.newVelocity.y -  (thisCell.pressure - downCell.pressure);
			thisCell.newVelocity.y = uY;
			thisCell.hasVelocityY = true;
		}
	}
	if (z > 0) {
		Cell3D& backCell = get3D(cells, x, y, z - 1);
		if (thisCell.content == CONTENT_FLUID || backCell.content == CONTENT_FLUID) {
			float uZ = thisCell.newVelocity.z -  (thisCell.pressure - backCell.pressure);
			thisCell.newVelocity.z = uZ;
			thisCell.hasVelocityZ = true;
		}
	}
}


__global__  void extrapolateVelocityByOne(Cell3D* cells, int sizeX, int sizeY, int sizeZ) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= sizeX * sizeY * sizeZ) return;

	int x = index / (sizeY * sizeZ);
	int y = (index - x * (sizeY * sizeZ)) / sizeZ;
	int z = index - x * (sizeY * sizeZ) - y * (sizeZ);

	Cell3D& thisCell = get3D(cells, x, y, z);
	const float epsilon = -1 + 1e-6;
	if (!thisCell.hasVelocityX) {
		float sumNeighborX = 0;
		int neighborXCount = 0;
		if (x > 0) {
			Cell3D& leftCell = get3D(cells, x - 1, y, z);
			if (leftCell.hasVelocityX && leftCell.newVelocity.x > epsilon) {
				sumNeighborX += leftCell.newVelocity.x;
				neighborXCount++;
			}
		}
		if (y > 0) {
			Cell3D& downCell = get3D(cells, x, y - 1, z);
			if (downCell.hasVelocityX && downCell.newVelocity.y > epsilon) {
				sumNeighborX += downCell.newVelocity.x;
				neighborXCount++;
			}
		}
		if (z > 0) {
			Cell3D& backCell = get3D(cells, x, y, z - 1);
			if (backCell.hasVelocityX && backCell.newVelocity.z > epsilon) {
				sumNeighborX += backCell.newVelocity.x;
				neighborXCount++;
			}
		}
		if (x < sizeX - 1) {
			Cell3D& rightCell = get3D(cells, x + 1, y, z);
			if (rightCell.hasVelocityX && rightCell.newVelocity.x < -epsilon) {
				sumNeighborX += rightCell.newVelocity.x;
				neighborXCount++;
			}
		}
		if (y < sizeY - 1) {
			Cell3D& upCell = get3D(cells, x, y + 1, z);
			if (upCell.hasVelocityX && upCell.newVelocity.y < -epsilon) {
				sumNeighborX += upCell.newVelocity.x;
				neighborXCount++;
			}
		}
		if (z < sizeZ - 1) {
			Cell3D& frontCell = get3D(cells, x, y, z + 1);
			if (frontCell.hasVelocityX && frontCell.newVelocity.z < -epsilon) {
				sumNeighborX += frontCell.newVelocity.x;
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
			Cell3D& leftCell = get3D(cells, x - 1, y, z);
			if (leftCell.hasVelocityY && leftCell.newVelocity.x > epsilon) {
				sumNeighborY += leftCell.newVelocity.y;
				neighborYCount++;
			}
		}
		if (y > 0) {
			Cell3D& downCell = get3D(cells, x, y - 1, z);
			if (downCell.hasVelocityY && downCell.newVelocity.y > epsilon) {
				sumNeighborY += downCell.newVelocity.y;
				neighborYCount++;
			}
		}
		if (z > 0) {
			Cell3D& backCell = get3D(cells, x, y, z - 1);
			if (backCell.hasVelocityY && backCell.newVelocity.z > epsilon) {
				sumNeighborY += backCell.newVelocity.y;
				neighborYCount++;
			}
		}
		if (x < sizeX - 1) {
			Cell3D& rightCell = get3D(cells, x + 1, y, z);
			if (rightCell.hasVelocityY && rightCell.newVelocity.x < -epsilon) {
				sumNeighborY += rightCell.newVelocity.y;
				neighborYCount++;
			}
		}
		if (y < sizeY - 1) {
			Cell3D& upCell = get3D(cells, x, y + 1, z);
			if (upCell.hasVelocityY && upCell.newVelocity.y < -epsilon) {
				sumNeighborY += upCell.newVelocity.y;
				neighborYCount++;
			}
		}
		if (z < sizeZ - 1) {
			Cell3D& frontCell = get3D(cells, x, y, z + 1);
			if (frontCell.hasVelocityY && frontCell.newVelocity.z < -epsilon) {
				sumNeighborY += frontCell.newVelocity.y;
				neighborYCount++;
			}
		}
		if (neighborYCount > 0) {
			thisCell.newVelocity.y = sumNeighborY / (float)neighborYCount;
			thisCell.hasVelocityY = true;
		}
	}

	if (!thisCell.hasVelocityZ) {
		float sumNeighborZ = 0;
		int neighborZCount = 0;
		if (x > 0) {
			Cell3D& leftCell = get3D(cells, x - 1, y, z);
			if (leftCell.hasVelocityZ && leftCell.newVelocity.x > epsilon) {
				sumNeighborZ += leftCell.newVelocity.z;
				neighborZCount++;
			}
		}
		if (y > 0) {
			Cell3D& downCell = get3D(cells, x, y - 1, z);
			if (downCell.hasVelocityZ && downCell.newVelocity.y > epsilon) {
				sumNeighborZ += downCell.newVelocity.z;
				neighborZCount++;
			}
		}
		if (z > 0) {
			Cell3D& backCell = get3D(cells, x, y, z - 1);
			if (backCell.hasVelocityZ && backCell.newVelocity.z > epsilon) {
				sumNeighborZ += backCell.newVelocity.z;
				neighborZCount++;
			}
		}
		if (x < sizeX - 1) {
			Cell3D& rightCell = get3D(cells, x + 1, y, z);
			if (rightCell.hasVelocityZ && rightCell.newVelocity.x < -epsilon) {
				sumNeighborZ += rightCell.newVelocity.z;
				neighborZCount++;
			}
		}
		if (y < sizeY - 1) {
			Cell3D& upCell = get3D(cells, x, y + 1, z);
			if (upCell.hasVelocityZ && upCell.newVelocity.y < -epsilon) {
				sumNeighborZ += upCell.newVelocity.z;
				neighborZCount++;
			}
		}
		if (z < sizeZ - 1) {
			Cell3D& frontCell = get3D(cells, x, y, z + 1);
			if (frontCell.hasVelocityZ && frontCell.newVelocity.z < -epsilon) {
				sumNeighborZ += frontCell.newVelocity.z;
				neighborZCount++;
			}
		}
		if (neighborZCount > 0) {
			thisCell.newVelocity.z = sumNeighborZ / (float)neighborZCount;
			thisCell.hasVelocityZ = true;
		}
	}
}



__global__  void computeDivergenceImpl(Cell3D* cells, int sizeX, int sizeY, int sizeZ, float cellPhysicalSize, float restParticlesPerCell) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= sizeX * sizeY * sizeZ) return;

	int x = index / (sizeY * sizeZ);
	int y = (index - x * (sizeY * sizeZ)) / sizeZ;
	int z = index - x * (sizeY * sizeZ) - y * (sizeZ);

	Cell3D& thisCell = get3D(cells, x, y, z);
	Cell3D& upCell = get3D(cells, x, y + 1, z);
	Cell3D& rightCell = get3D(cells, x + 1, y, z);
	Cell3D& frontCell = get3D(cells, x, y, z + 1);

	float div = (upCell.newVelocity.y - thisCell.newVelocity.y + rightCell.newVelocity.x - thisCell.newVelocity.x + frontCell.newVelocity.z - thisCell.newVelocity.z);

	//div -= max((thisCell.density - restParticlesPerCell) * 1.0, 0.0); //volume conservation
	//div -= (thisCell.density - restParticlesPerCell) * 1.0; //volume conservation

	if (thisCell.density > restParticlesPerCell) {
		div -= (thisCell.density - restParticlesPerCell) * 0.01;
	}
	else if (thisCell.density <= restParticlesPerCell) {
		div -= (thisCell.density - restParticlesPerCell) * 0.01;
	}

	thisCell.divergence = div;

}
__global__  void resetPressureImpl(Cell3D* cells, int sizeX, int sizeY, int sizeZ) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= sizeX * sizeY * sizeZ) return;

	int x = index / (sizeY * sizeZ);
	int y = (index - x * (sizeY * sizeZ)) / sizeZ;
	int z = index - x * (sizeY * sizeZ) - y * (sizeZ);

	Cell3D& thisCell = get3D(cells, x, y, z);
	thisCell.pressure = 0;

}

__global__  void precomputeNeighbors(Cell3D* cells, int sizeX, int sizeY, int sizeZ) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= sizeX * sizeY * sizeZ) return;

	int x = index / (sizeY * sizeZ);
	int y = (index - x * (sizeY * sizeZ)) / sizeZ;
	int z = index - x * (sizeY * sizeZ) - y * (sizeZ);

	Cell3D& thisCell = get3D(cells, x, y, z);
	thisCell.leftCell = &thisCell;
	thisCell.rightCell = &thisCell;
	thisCell.upCell = &thisCell;
	thisCell.downCell = &thisCell;
	thisCell.frontCell = &thisCell;
	thisCell.backCell = &thisCell;

	if (x > 0) {
		thisCell.leftCell = &get3D(cells, x-1, y, z);
	}
	if (y > 0) {
		thisCell.downCell = &get3D(cells, x, y-1, z);
	}
	if (z > 0) {
		thisCell.backCell = &get3D(cells, x, y , z-1);
	}

	if (x < sizeX-1) {
		thisCell.rightCell = &get3D(cells, x + 1, y, z);
	}
	if (y < sizeY - 1) {
		thisCell.upCell = &get3D(cells, x , y+1, z);
	}
	if (z < sizeZ - 1) {
		thisCell.frontCell = &get3D(cells, x , y, z+1);
	}
}

__global__  void jacobiImpl(Cell3D* cells, int sizeX, int sizeY, int sizeZ, float cellPhysicalSize) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= sizeX * sizeY * sizeZ) return;

	int x = index / (sizeY * sizeZ);
	int y = (index - x * (sizeY * sizeZ)) / sizeZ;
	int z = index - x * (sizeY * sizeZ) - y * (sizeZ);

	Cell3D& thisCell = get3D(cells, x, y, z);

	if (thisCell.content == CONTENT_AIR) {
		thisCell.pressure = 0;
		return;
	}

	float RHS = -thisCell.divergence;

	float newPressure = 0;

	float centerCoeff = 6;

	newPressure += thisCell.leftCell->pressure;
	newPressure += thisCell.rightCell->pressure;
	newPressure += thisCell.upCell->pressure;
	newPressure += thisCell.downCell->pressure; 
	newPressure += thisCell.frontCell->pressure;
	newPressure += thisCell.backCell->pressure;

	newPressure += RHS;
	newPressure /= centerCoeff;



	thisCell.pressure = newPressure;
}


__global__  void writeIndicesImpl(int* particleIndices, int particleCount) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= particleCount) return;

	particleIndices[index] = index;
}



