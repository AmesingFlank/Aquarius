#include "Fluid_3D_kernels.cuh"


__global__  void applyGravityImpl(VolumeCollection volumes, int sizeX, int sizeY, int sizeZ, float timeStep, float3 gravitationalAcceleration) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= sizeX || y >= sizeY || z >= sizeZ) return;

	//gravitationalAcceleration = 0;

	if (volumes.content.readSurface<int>(x,y,z) == CONTENT_FLUID) {
		float4 newVelocity = volumes.newVelocity.readSurface<float4>(x, y, z);
		newVelocity += make_float4(gravitationalAcceleration * timeStep,0);


		volumes.newVelocity.writeSurface<float4>(newVelocity, x, y, z);

		if (volumes.content.readSurface<int>(x+1, y, z) == CONTENT_AIR && gravitationalAcceleration.x != 0) {
			float4 newVelocity = volumes.newVelocity.readSurface<float4>(x+1, y, z);
			newVelocity += make_float4(gravitationalAcceleration * timeStep, 0);
			volumes.newVelocity.writeSurface<float4>(newVelocity, x+1, y, z);

		}
		if (volumes.content.readSurface<int>(x, y + 1, z) == CONTENT_AIR && gravitationalAcceleration.y != 0) {
			float4 newVelocity = volumes.newVelocity.readSurface<float4>(x, y + 1, z);
			newVelocity += make_float4(gravitationalAcceleration * timeStep, 0);
			volumes.newVelocity.writeSurface<float4>(newVelocity, x, y + 1, z);

		}
		if (volumes.content.readSurface<int>(x, y, z+1) == CONTENT_AIR && gravitationalAcceleration.z != 0) {
			float4 newVelocity = volumes.newVelocity.readSurface<float4>(x, y, z+1);
			newVelocity += make_float4(gravitationalAcceleration * timeStep, 0);
			volumes.newVelocity.writeSurface<float4>(newVelocity, x, y , z+1);

		}
	}
}




__global__  void fixBoundaryX(VolumeCollection volumes, int sizeX, int sizeY, int sizeZ) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= (sizeY + 1) * (sizeZ + 1)) return;

	int y = index / (sizeZ + 1);
	int z = index - y * (sizeZ + 1);

	float4 newVelocity0 = volumes.newVelocity.readSurface<float4>(0,y, z);
	newVelocity0.x = max(0.f, newVelocity0.x);
	volumes.newVelocity.writeSurface<float4>(newVelocity0, 0,y, z);

	float4 newVelocity1 = volumes.newVelocity.readSurface<float4>(sizeX,y, z);
	newVelocity1.x = min(0.f, newVelocity1.x);
	volumes.newVelocity.writeSurface<float4>(newVelocity1, sizeX,y, z);

	volumes.content.writeSurface<int>(CONTENT_SOLID, sizeX, y, z);

}

__global__  void fixBoundaryY(VolumeCollection volumes, int sizeX, int sizeY, int sizeZ) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= (sizeX + 1) * (sizeZ + 1)) return;

	int x = index / (sizeZ + 1);
	int z = index - x * (sizeZ + 1);

	float4 newVelocity0 = volumes.newVelocity.readSurface<float4>(x, 0, z);
	newVelocity0.y = max(0.f,newVelocity0.y);
	volumes.newVelocity.writeSurface<float4>(newVelocity0, x, 0, z);

	float4 newVelocity1 = volumes.newVelocity.readSurface<float4>(x, sizeY, z);
	newVelocity1.y = min(0.f, newVelocity1.y);
	volumes.newVelocity.writeSurface<float4>(newVelocity1, x, sizeY, z);


	volumes.content.writeSurface<int>(CONTENT_SOLID, x, sizeY, z);
}

__global__  void fixBoundaryZ(VolumeCollection volumes, int sizeX, int sizeY, int sizeZ) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= (sizeX + 1) * (sizeY + 1)) return;

	int x = index / (sizeY + 1);
	int y = index - x * (sizeY + 1);

	float4 newVelocity0 = volumes.newVelocity.readSurface<float4>(x, y, 0);
	newVelocity0.z = max(0.f, newVelocity0.z);;
	volumes.newVelocity.writeSurface<float4>(newVelocity0, x, y,0);

	float4 newVelocity1 = volumes.newVelocity.readSurface<float4>(x, y,sizeZ);
	newVelocity1.z = min(0.f, newVelocity1.z);
	volumes.newVelocity.writeSurface<float4>(newVelocity1, x, y,sizeZ);

	volumes.content.writeSurface<int>(CONTENT_SOLID, x, y, sizeZ);
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
	float thisDensity = volumes.density.readSurface<float>(x, y, z);

	float4 thisNewVelocity = volumes.newVelocity.readSurface<float4>(x, y, z);

	float densityUsed; // technically, a MAC descritization of the density field should be used. This here is an approximation


	if (x > 0) {
		
		int leftContent = volumes.content.readSurface<int>(x-1, y, z);
		float leftPressure = volumes.pressure.readSurface<float>(x-1, y, z);

		if (thisCellContent == CONTENT_FLUID || leftContent == CONTENT_FLUID) {
			densityUsed = thisDensity;
			float leftDensity = volumes.density.readSurface<float>(x-1, y, z);
			if (thisCellContent == CONTENT_FLUID &&  leftContent == CONTENT_FLUID) {
				densityUsed = (leftDensity + thisDensity) / 2;
			}
			else if(leftContent == CONTENT_FLUID){
				densityUsed = leftDensity;
			}


			float uX = thisNewVelocity.x -  (thisPressure - leftPressure) / densityUsed;
			thisNewVelocity.x = uX;
			hasVelocity.x = true;
		}
	}
	if (y > 0) {
		
		int downContent = volumes.content.readSurface<int>(x, y-1, z);
		float downPressure = volumes.pressure.readSurface<float>(x, y-1, z);
		if (thisCellContent == CONTENT_FLUID || downContent == CONTENT_FLUID) {

			densityUsed = thisDensity;
			float downDensity = volumes.density.readSurface<float>(x, y-1, z);
			if (thisCellContent == CONTENT_FLUID && downContent == CONTENT_FLUID) {
				densityUsed = (downDensity + thisDensity) / 2;
			}
			else if (downContent == CONTENT_FLUID) {
				densityUsed = downDensity;
			}

			float uY = thisNewVelocity.y -  (thisPressure - downPressure) / densityUsed;
			thisNewVelocity.y = uY;
			hasVelocity.y = true;
		}
	}
	if (z > 0) {
		
		int backContent = volumes.content.readSurface<int>(x, y, z-1);
		float backPressure = volumes.pressure.readSurface<float>(x, y, z-1);

		if (thisCellContent == CONTENT_FLUID || backContent == CONTENT_FLUID) {

			densityUsed = thisDensity;
			float backDensity = volumes.density.readSurface<float>(x, y, z-1);
			if (thisCellContent == CONTENT_FLUID && backContent == CONTENT_FLUID) {
				densityUsed = (backDensity + thisDensity) / 2;
			}
			else if (backContent == CONTENT_FLUID) {
				densityUsed = backDensity;

			}

			


			float uZ = thisNewVelocity.z -  (thisPressure  - backPressure) / densityUsed;
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
			float4 newVelocity = volumes.newVelocity.readSurface<float4>(x, y+1, z);
			if (upHasVelocity.x && newVelocity.y < -epsilon) {
				sumNeighborX += newVelocity.x;
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
			float4 newVelocity = volumes.newVelocity.readSurface<float4>(x, y + 1, z);
			if (upHasVelocity.y && newVelocity.y < -epsilon) {
				sumNeighborY += newVelocity.y;
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
			float4 newVelocity = volumes.newVelocity.readSurface<float4>(x, y + 1, z);
			if (upHasVelocity.z && newVelocity.y < -epsilon) {
				sumNeighborZ += newVelocity.z;
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
	float4 newVelocity = volumes.newVelocity.readSurface<float4>(x, y + 1, z);
	float4 rightNewVelocity = volumes.newVelocity.readSurface<float4>(x + 1, y, z);
	float4 frontNewVelocity = volumes.newVelocity.readSurface<float4>(x, y, z + 1);

	float div = (newVelocity.y - thisNewVelocity.y + rightNewVelocity.x - thisNewVelocity.x + frontNewVelocity.z - thisNewVelocity.z);

	//div -= max((thisCell.density - restParticlesPerCell) * 1.0, 0.0); //volume conservation
	//div -= (thisCell.density - restParticlesPerCell) * 1.0; //volume conservation

	int currentCount = volumes.particleCount.readSurface<int>(x,y,z);

	if (currentCount > restParticlesPerCell) {
		div -= (currentCount - restParticlesPerCell) * 0.001;
	}
	else if (currentCount <= restParticlesPerCell) {
		div -= (currentCount - restParticlesPerCell) * 0.001;
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

	float thisDensity = volumes.density.readTexture<float>(x, y, z);
	// technically, a MAC descritization of the density field should be used. This here is an approximation



	float RHS = -volumes.divergence.readSurface<float>(x, y, z)*thisDensity;

	float newPressure = 0;

	

	float centerCoeff = 6 ;


	newPressure += volumes.pressure.readTexture<float>(x+1, y, z) ;
	newPressure += volumes.pressure.readTexture<float>(x-1, y, z) ;
	newPressure += volumes.pressure.readTexture<float>(x, y+1, z) ;
	newPressure += volumes.pressure.readTexture<float>(x, y-1, z) ;
	newPressure += volumes.pressure.readTexture<float>(x, y, z+1) ;
	newPressure += volumes.pressure.readTexture<float>(x, y, z-1) ;


	newPressure += RHS;
	newPressure /= centerCoeff;

	if (x == 25 && z == 25 && y == 2) {
		//printf("%f\n", newPressure);
	}


	volumes.pressure.writeSurface<float>(newPressure, x, y, z);
}


__global__  void writeIndicesImpl(int* particleIndices, int particleCount) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= particleCount) return;

	particleIndices[index] = index;
}





__global__  void diffusionJacobiImpl(VolumeCollection volumes, int sizeX, int sizeY, int sizeZ, float lambda) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= sizeX || y >= sizeY || z >= sizeZ) return;



	if (volumes.content.readSurface<int>(x, y, z) == CONTENT_AIR) {
		volumes.pressure.writeSurface<float>(0.f, x, y, z);
		return;
	}

	float4 RHS = volumes.newVolumeFractions.readSurface<float4>(x, y, z);

	
	float4 result = make_float4(0, 0, 0, 0);
	float centerCoeff = 1.f+ 2*lambda;


	result += volumes.newVolumeFractions.readTexture<float4>(x + 1, y, z)*lambda;
	result += volumes.newVolumeFractions.readTexture<float4>(x - 1, y, z)*lambda;
	result += volumes.newVolumeFractions.readTexture<float4>(x, y + 1, z)*lambda;
	result += volumes.newVolumeFractions.readTexture<float4>(x, y - 1, z)*lambda;
	result += volumes.newVolumeFractions.readTexture<float4>(x, y, z + 1)*lambda;
	result += volumes.newVolumeFractions.readTexture<float4>(x, y, z - 1)*lambda;


	result += RHS;
	result /= centerCoeff;
	result /= (result.x + result.y + result.z + result.w);


	volumes.newVolumeFractions.writeSurface<float4>(result, x, y, z);
	
}
