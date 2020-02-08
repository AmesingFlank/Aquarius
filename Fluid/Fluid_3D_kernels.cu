#include "Fluid_3D_kernels.cuh"



__global__  void applyGravityImpl(VolumeCollection volumes, int sizeX, int sizeY, int sizeZ, float timeStep, float gravitationalAcceleration) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= sizeX || y >= sizeY || z >= sizeZ) return;

	if (volumes.content.readSurface<int>(x,y,z) == CONTENT_FLUID) {
		float4 newVelocity = volumes.newVelocity.readSurface<float4>(x, y, z);
		newVelocity.y -= gravitationalAcceleration * timeStep;


		volumes.newVelocity.writeSurface<float4>(newVelocity, x, y, z);

		if (volumes.content.readSurface<int>(x, y+1, z) == CONTENT_AIR) {
			float4 upNewVelocity = volumes.newVelocity.readSurface<float4>(x, y+1, z);
			upNewVelocity.y -= gravitationalAcceleration * timeStep;
			volumes.newVelocity.writeSurface<float4>(upNewVelocity, x, y+1, z);

		}
	}
}




__global__  void fixBoundaryX(VolumeCollection volumes, int sizeX, int sizeY, int sizeZ) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= (sizeY + 1) * (sizeZ + 1)) return;

	int y = index / (sizeZ + 1);
	int z = index - y * (sizeZ + 1);

	float4 newVelocity0 = volumes.newVelocity.readSurface<float4>(0,y, z);
	newVelocity0.x = 0;
	volumes.newVelocity.writeSurface<float4>(newVelocity0, 0,y, z);

	float4 upNewVelocity1 = volumes.newVelocity.readSurface<float4>(sizeX,y, z);
	upNewVelocity1.x = 0;
	volumes.newVelocity.writeSurface<float4>(upNewVelocity1, sizeX,y, z);

	volumes.content.writeSurface<int>(CONTENT_SOLID, sizeX, y, z);

}

__global__  void fixBoundaryY(VolumeCollection volumes, int sizeX, int sizeY, int sizeZ) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= (sizeX + 1) * (sizeZ + 1)) return;

	int x = index / (sizeZ + 1);
	int z = index - x * (sizeZ + 1);

	float4 newVelocity0 = volumes.newVelocity.readSurface<float4>(x, 0, z);
	newVelocity0.y = 0;
	volumes.newVelocity.writeSurface<float4>(newVelocity0, x, 0, z);

	float4 upNewVelocity1 = volumes.newVelocity.readSurface<float4>(x, sizeY, z);
	upNewVelocity1.y = 0;
	volumes.newVelocity.writeSurface<float4>(upNewVelocity1, x, sizeY, z);


	volumes.content.writeSurface<int>(CONTENT_SOLID, x, sizeY, z);
}

__global__  void fixBoundaryZ(VolumeCollection volumes, int sizeX, int sizeY, int sizeZ) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= (sizeX + 1) * (sizeY + 1)) return;

	int x = index / (sizeY + 1);
	int y = index - x * (sizeY + 1);

	float4 newVelocity0 = volumes.newVelocity.readSurface<float4>(x, y, 0);
	newVelocity0.z = 0;
	volumes.newVelocity.writeSurface<float4>(newVelocity0, x, y,0);

	float4 upNewVelocity1 = volumes.newVelocity.readSurface<float4>(x, y,sizeZ);
	upNewVelocity1.z = 0;
	volumes.newVelocity.writeSurface<float4>(upNewVelocity1, x, y,sizeZ);

	volumes.content.writeSurface<int>(CONTENT_SOLID, x, y, sizeZ);
}


__device__ float getNeibourCoefficient(int x, int y, int z, float u, float& centerCoefficient, float& RHS, VolumeCollection volumes,int sizeX, int sizeY, int sizeZ) {

	int neibourContent = volumes.content.readSurface<int>(x, y, z);

	if (x >= 0 && x < sizeX && y >= 0 && y < sizeY && z >= 0 && z < sizeZ && 
		neibourContent == CONTENT_FLUID) {
		return -1;
	}
	else {
		if (x < 0 || y < 0 || z < 0 || x >= sizeX || y >= sizeY || z >= sizeZ || 
			neibourContent == CONTENT_SOLID) {
			centerCoefficient -= 1;
			//RHS += u;
			return 0;
		}
		else if (neibourContent == CONTENT_AIR) {
			return 0;
		}
	}
}



__global__  void constructPressureEquations(VolumeCollection volumes, int sizeX, int sizeY, int sizeZ, PressureEquation3D* equations, bool* hasNonZeroRHS) {


	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= sizeX || y >= sizeY || z >= sizeZ) return;

	volumes.pressure.writeSurface<float>(0.f, x, y, z);
	
	if (volumes.content.readSurface<int>(x,y,z) != CONTENT_FLUID)
		return;

	PressureEquation3D thisEquation;
	float RHS = -volumes.divergence.readSurface<float>(x, y, z);

	float centerCoeff = 6;

	float4 thisNewVelocity = volumes.newVelocity.readSurface<float4>(x, y, z);
	float4 rightNewVelocity = volumes.newVelocity.readSurface<float4>(x+1, y, z);
	float4 upNewVelocity = volumes.newVelocity.readSurface<float4>(x, y+1, z);
	float4 frontNewVelocity = volumes.newVelocity.readSurface<float4>(x, y, z+1);

	float leftCoeff = getNeibourCoefficient(x - 1, y, z, thisNewVelocity.x, centerCoeff, RHS,volumes, sizeX, sizeY, sizeZ);
	float rightCoeff = getNeibourCoefficient(x + 1, y, z, rightNewVelocity.x, centerCoeff, RHS, volumes, sizeX, sizeY, sizeZ);
	float downCoeff = getNeibourCoefficient(x, y - 1, z, thisNewVelocity.y, centerCoeff, RHS, volumes, sizeX, sizeY, sizeZ);
	float upCoeff = getNeibourCoefficient(x, y + 1, z,  upNewVelocity.y, centerCoeff, RHS, volumes, sizeX, sizeY, sizeZ);
	float backCoeff = getNeibourCoefficient(x, y, z - 1, thisNewVelocity.z, centerCoeff, RHS, volumes, sizeX, sizeY, sizeZ);
	float frontCoeff = getNeibourCoefficient(x, y, z + 1,  frontNewVelocity.z, centerCoeff, RHS, volumes, sizeX, sizeY, sizeZ);

	int nnz = 0;

	if (downCoeff) {
		int downIndex = volumes.fluidIndex.readSurface<int>(x, y - 1, z);
		thisEquation.termsIndex[thisEquation.termCount] = downIndex;
		thisEquation.termsCoeff[thisEquation.termCount] = downCoeff;
		++thisEquation.termCount;
		++nnz;
	}
	if (leftCoeff) {
		int leftIndex = volumes.fluidIndex.readSurface<int>(x-1, y, z);
		thisEquation.termsIndex[thisEquation.termCount] = leftIndex;
		thisEquation.termsCoeff[thisEquation.termCount] = leftCoeff;
		++thisEquation.termCount;
		++nnz;
	}
	if (backCoeff) {
		int backIndex = volumes.fluidIndex.readSurface<int>(x, y ,z-1);
		thisEquation.termsIndex[thisEquation.termCount] = backIndex;
		thisEquation.termsCoeff[thisEquation.termCount] = backCoeff;
		++thisEquation.termCount;
		++nnz;
	}
	int thisIndex = volumes.fluidIndex.readSurface<int>(x, y, z );
	thisEquation.termsIndex[thisEquation.termCount] = thisIndex;
	thisEquation.termsCoeff[thisEquation.termCount] = centerCoeff;
	++thisEquation.termCount;
	if (rightCoeff) {
		int rightIndex = volumes.fluidIndex.readSurface<int>(x+1, y, z);
		thisEquation.termsIndex[thisEquation.termCount] = rightIndex;
		thisEquation.termsCoeff[thisEquation.termCount] = rightCoeff;
		++thisEquation.termCount;
		++nnz;
	}
	if (upCoeff) {
		int upIndex = volumes.fluidIndex.readSurface<int>(x, y+1, z);
		thisEquation.termsIndex[thisEquation.termCount] = upIndex;
		thisEquation.termsCoeff[thisEquation.termCount] = upCoeff;
		++thisEquation.termCount;
		++nnz;
	}
	if (frontCoeff) {
		int frontIndex = volumes.fluidIndex.readSurface<int>(x, y, z+1);
		thisEquation.termsIndex[thisEquation.termCount] = frontIndex;
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
	equations[thisIndex] = thisEquation;

}

__global__  void setPressure(VolumeCollection volumes, int sizeX, int sizeY, int sizeZ, double* pressureResult) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= sizeX || y >= sizeY || z >= sizeZ) return;

	if (volumes.content.readSurface<int>(x, y, z) != CONTENT_FLUID)
		return;

	int thisIndex = volumes.fluidIndex.readSurface<int>(x, y, z);

	volumes.pressure.writeSurface<float>(pressureResult[thisIndex], x, y, z);

}


__global__  void updateVelocityWithPressureImpl(VolumeCollection volumes, int sizeX, int sizeY, int sizeZ) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= sizeX || y >= sizeY || z >= sizeZ) return;

	

	int4 hasVelocity = make_int4(0, 0, 0, 0);

	int thisCellContent = volumes.content.readSurface<int>(x, y, z);
	float thisPressure = volumes.pressure.readSurface<float>(x, y, z);

	float4 thisNewVelocity = volumes.newVelocity.readSurface<float4>(x, y, z);

	if (x > 0) {
		

		int leftContent = volumes.content.readSurface<int>(x-1, y, z);
		float leftPressure = volumes.pressure.readSurface<float>(x-1, y, z);
		
		if (thisCellContent == CONTENT_FLUID || leftContent == CONTENT_FLUID) {
			float uX = thisNewVelocity.x -  (thisPressure - leftPressure);
			thisNewVelocity.x = uX;
			hasVelocity.x = true;
		}
	}
	if (y > 0) {
		
		int downContent = volumes.content.readSurface<int>(x, y-1, z);
		float downPressure = volumes.pressure.readSurface<float>(x, y-1, z);
		if (thisCellContent == CONTENT_FLUID || downContent == CONTENT_FLUID) {
			float uY = thisNewVelocity.y -  (thisPressure - downPressure);
			thisNewVelocity.y = uY;
			hasVelocity.x = true;
		}
	}
	if (z > 0) {
		
		int backContent = volumes.content.readSurface<int>(x, y, z-1);
		float backPressure = volumes.pressure.readSurface<float>(x, y, z-1);
		if (thisCellContent == CONTENT_FLUID || backContent == CONTENT_FLUID) {
			float uZ = thisNewVelocity.z -  (thisPressure - backPressure);
			thisNewVelocity.z = uZ;
			hasVelocity.z = true;
		}
	}

	volumes.hasVelocity.writeSurface<int4>(hasVelocity, x, y, z);
	volumes.newVelocity.writeSurface<float4>(thisNewVelocity, x, y, z);
}


__global__  void extrapolateVelocityByOne(VolumeCollection volumes, int sizeX, int sizeY, int sizeZ) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= sizeX || y >= sizeY || z >= sizeZ) return;


	const float epsilon = -1 + 1e-6;

	int4 thisHasVelocity = volumes.hasVelocity.readSurface<int4>(x, y, z);
	float4 thisNewVelocity = volumes.newVelocity.readSurface<float4>(x, y, z);
	
	


	if (!thisHasVelocity.x) {
		float sumNeighborX = 0;
		int neighborXCount = 0;
		if (x > 0) {
			
			int4 leftHasVelocity = volumes.hasVelocity.readSurface<int4>(x - 1, y, z);
			float4 leftNewVelocity = volumes.newVelocity.readSurface<float4>(x - 1, y, z);
			if (leftHasVelocity.x && leftNewVelocity.x > epsilon) {
				sumNeighborX += leftNewVelocity.x;
				neighborXCount++;
			}
		}
		if (y > 0) {
			
			int4 downHasVelocity = volumes.hasVelocity.readSurface<int4>(x, y - 1, z);
			float4 downNewVelocity = volumes.newVelocity.readSurface<float4>(x , y-1, z);
			if (downHasVelocity.x && downNewVelocity.y > epsilon) {
				sumNeighborX += downNewVelocity.x;
				neighborXCount++;
			}
		}
		if (z > 0) {
			
			int4 backHasVelocity = volumes.hasVelocity.readSurface<int4>(x, y, z-1);
			float4 backNewVelocity = volumes.newVelocity.readSurface<float4>(x, y , z-1);
			if (backHasVelocity.x && backNewVelocity.z > epsilon) {
				sumNeighborX += backNewVelocity.x;
				neighborXCount++;
			}
		}
		if (x < sizeX - 1) {
			
			int4 rightHasVelocity = volumes.hasVelocity.readSurface<int4>(x + 1, y, z);
			float4 rightNewVelocity = volumes.newVelocity.readSurface<float4>(x + 1, y, z);
			if (rightHasVelocity.x && rightNewVelocity.x < -epsilon) {
				sumNeighborX += rightNewVelocity.x;
				neighborXCount++;
			}
		}
		if (y < sizeY - 1) {
			
			int4 upHasVelocity = volumes.hasVelocity.readSurface<int4>(x, y+1, z);
			float4 upNewVelocity = volumes.newVelocity.readSurface<float4>(x, y+1, z);
			if (upHasVelocity.x && upNewVelocity.y < -epsilon) {
				sumNeighborX += upNewVelocity.x;
				neighborXCount++;
			}
		}
		if (z < sizeZ - 1) {
			
			int4 frontHasVelocity = volumes.hasVelocity.readSurface<int4>(x, y, z + 1);
			float4 frontNewVelocity = volumes.newVelocity.readSurface<float4>(x, y, z + 1);
			if (frontHasVelocity.x && frontNewVelocity.z < -epsilon) {
				sumNeighborX += frontNewVelocity.x;
				neighborXCount++;
			}
		}

		if (neighborXCount > 0) {
			thisNewVelocity.x = sumNeighborX / (float)neighborXCount;
			thisHasVelocity.x = true;
		}
	}

	if (!thisHasVelocity.y) {
		float sumNeighborY = 0;
		int neighborYCount = 0;
		if (x > 0) {
			
			int4 leftHasVelocity = volumes.hasVelocity.readSurface<int4>(x - 1, y, z);
			float4 leftNewVelocity = volumes.newVelocity.readSurface<float4>(x - 1, y, z);
			if (leftHasVelocity.y && leftNewVelocity.x > epsilon) {
				sumNeighborY += leftNewVelocity.y;
				neighborYCount++;
			}
		}
		if (y > 0) {
			
			float4 downNewVelocity = volumes.newVelocity.readSurface<float4>(x, y - 1, z);
			int4 downHasVelocity = volumes.hasVelocity.readSurface<int4>(x, y - 1, z);
			if (downHasVelocity.y && downNewVelocity.y > epsilon) {
				sumNeighborY += downNewVelocity.y;
				neighborYCount++;
			}
		}
		if (z > 0) {
			
			int4 backHasVelocity = volumes.hasVelocity.readSurface<int4>(x, y, z - 1);
			float4 backNewVelocity = volumes.newVelocity.readSurface<float4>(x, y, z - 1);
			if (backHasVelocity.y && backNewVelocity.z > epsilon) {
				sumNeighborY += backNewVelocity.y;
				neighborYCount++;
			}
		}
		if (x < sizeX - 1) {
			
			int4 rightHasVelocity = volumes.hasVelocity.readSurface<int4>(x + 1, y, z);
			float4 rightNewVelocity = volumes.newVelocity.readSurface<float4>(x + 1, y, z);
			if (rightHasVelocity.y && rightNewVelocity.x < -epsilon) {
				sumNeighborY += rightNewVelocity.y;
				neighborYCount++;
			}
		}
		if (y < sizeY - 1) {
			
			int4 upHasVelocity = volumes.hasVelocity.readSurface<int4>(x, y + 1, z);
			float4 upNewVelocity = volumes.newVelocity.readSurface<float4>(x, y + 1, z);
			if (upHasVelocity.y && upNewVelocity.y < -epsilon) {
				sumNeighborY += upNewVelocity.y;
				neighborYCount++;
			}
		}
		if (z < sizeZ - 1) {
			
			int4 frontHasVelocity = volumes.hasVelocity.readSurface<int4>(x, y, z + 1);
			float4 frontNewVelocity = volumes.newVelocity.readSurface<float4>(x, y, z + 1);
			if (frontHasVelocity.y && frontNewVelocity.z < -epsilon) {
				sumNeighborY += frontNewVelocity.y;
				neighborYCount++;
			}
		}
		if (neighborYCount > 0) {
			thisNewVelocity.y = sumNeighborY / (float)neighborYCount;
			thisHasVelocity.y = true;
		}
	}

	if (!thisHasVelocity.z) {
		float sumNeighborZ = 0;
		int neighborZCount = 0;
		if (x > 0) {
			
			int4 leftHasVelocity = volumes.hasVelocity.readSurface<int4>(x - 1, y, z);
			float4 leftNewVelocity = volumes.newVelocity.readSurface<float4>(x - 1, y, z);
			if (leftHasVelocity.z && leftNewVelocity.x > epsilon) {
				sumNeighborZ += leftNewVelocity.z;
				neighborZCount++;
			}
		}
		if (y > 0) {
			
			float4 downNewVelocity = volumes.newVelocity.readSurface<float4>(x, y - 1, z);
			int4 downHasVelocity = volumes.hasVelocity.readSurface<int4>(x, y-1, z);
			if (downHasVelocity.z && downNewVelocity.y > epsilon) {
				sumNeighborZ += downNewVelocity.z;
				neighborZCount++;
			}
		}
		if (z > 0) {
			
			int4 backHasVelocity = volumes.hasVelocity.readSurface<int4>(x, y, z - 1);
			float4 backNewVelocity = volumes.newVelocity.readSurface<float4>(x, y, z - 1);
			if (backHasVelocity.z && backNewVelocity.z > epsilon) {
				sumNeighborZ += backNewVelocity.z;
				neighborZCount++;
			}
		}
		if (x < sizeX - 1) {
			
			int4 rightHasVelocity = volumes.hasVelocity.readSurface<int4>(x+1, y, z );
			float4 rightNewVelocity = volumes.newVelocity.readSurface<float4>(x+1, y, z);
			if (rightHasVelocity.z && rightNewVelocity.x < -epsilon) {
				sumNeighborZ += rightNewVelocity.z;
				neighborZCount++;
			}
		}
		if (y < sizeY - 1) {
			
			int4 upHasVelocity = volumes.hasVelocity.readSurface<int4>(x, y + 1, z);
			float4 upNewVelocity = volumes.newVelocity.readSurface<float4>(x, y + 1, z);
			if (upHasVelocity.z && upNewVelocity.y < -epsilon) {
				sumNeighborZ += upNewVelocity.z;
				neighborZCount++;
			}
		}
		if (z < sizeZ - 1) {
			
			int4 frontHasVelocity = volumes.hasVelocity.readSurface<int4>(x, y, z+1);
			float4 frontNewVelocity = volumes.newVelocity.readSurface<float4>(x, y , z+1);
			if (frontHasVelocity.z && frontNewVelocity.z < -epsilon) {
				sumNeighborZ += frontNewVelocity.z;
				neighborZCount++;
			}
		}
		if (neighborZCount > 0) {
			thisNewVelocity.z = sumNeighborZ / (float)neighborZCount;
			thisHasVelocity.z = true;
		}
	}

	volumes.hasVelocity.writeSurface<int4>(thisHasVelocity, x, y, z);
	volumes.newVelocity.writeSurface<float4>(thisNewVelocity, x, y, z);
}



__global__  void computeDivergenceImpl(VolumeCollection volumes, int sizeX, int sizeY, int sizeZ, float cellPhysicalSize, float restParticlesPerCell) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= sizeX || y >= sizeY || z >= sizeZ) return;

	
	
	
	

	float4 thisNewVelocity = volumes.newVelocity.readSurface<float4>(x, y, z);
	float4 upNewVelocity = volumes.newVelocity.readSurface<float4>(x, y + 1, z);
	float4 rightNewVelocity = volumes.newVelocity.readSurface<float4>(x + 1, y, z);
	float4 frontNewVelocity = volumes.newVelocity.readSurface<float4>(x, y, z + 1);

	float div = (upNewVelocity.y - thisNewVelocity.y + rightNewVelocity.x - thisNewVelocity.x + frontNewVelocity.z - thisNewVelocity.z);

	//div -= max((thisCell.density - restParticlesPerCell) * 1.0, 0.0); //volume conservation
	//div -= (thisCell.density - restParticlesPerCell) * 1.0; //volume conservation

	int currentCount = volumes.particleCount.readSurface<int>(x,y,z);

	if (currentCount > restParticlesPerCell) {
		div -= (currentCount - restParticlesPerCell) * 0.01;
	}
	else if (currentCount <= restParticlesPerCell) {
		div -= (currentCount - restParticlesPerCell) * 0.01;
	}

	volumes.divergence.writeSurface<float>(div, x, y, z);
}

__global__  void jacobiImpl(VolumeCollection volumes, int sizeX, int sizeY, int sizeZ, float cellPhysicalSize) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= sizeX || y >= sizeY || z >= sizeZ) return;

	

	if (volumes.content.readSurface<int>(x,y,z) == CONTENT_AIR) {
		volumes.pressure.writeSurface<float>(0.f, x, y, z);
		return;
	}

	float RHS = -volumes.divergence.readSurface<float>(x, y, z);

	float newPressure = 0;

	float centerCoeff = 6;


	newPressure += volumes.pressure.readTexture<float>(x+1, y, z);
	newPressure += volumes.pressure.readTexture<float>(x-1, y, z);
	newPressure += volumes.pressure.readTexture<float>(x, y+1, z);
	newPressure += volumes.pressure.readTexture<float>(x, y-1, z);
	newPressure += volumes.pressure.readTexture<float>(x, y, z+1);
	newPressure += volumes.pressure.readTexture<float>(x, y, z-1);


	newPressure += RHS;
	newPressure /= centerCoeff;


	volumes.pressure.writeSurface<float>(newPressure, x, y, z);
}


__global__  void writeIndicesImpl(int* particleIndices, int particleCount) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= particleCount) return;

	particleIndices[index] = index;
}



