#include "MAC_Grid_3D.cuh"



__global__
void setFluidIndex(VolumeCollection volumes, int cellCount, unsigned int* fluidCount, int sizeX, int sizeY, int sizeZ) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x > sizeX || y > sizeY || z > sizeZ) return;

		
	if (volumes.content.readSurface<int>(x,y,z) == CONTENT_FLUID) {
		int thisIndex = atomicInc(fluidCount, cellCount);
		volumes.content.writeSurface<int>(thisIndex, x, y, z);
	}

}


	


__global__
void writeSpeed(VolumeCollection volumes, int cellCount, float* speedX, float* speedY, float* speedZ,int sizeX, int sizeY, int sizeZ) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x > sizeX || y > sizeY || z > sizeZ) return;
	float4 velocity = volumes.velocity.readSurface<float4>(x, y, z);

	get3D(speedX, x, y, z) = abs(velocity.x);
	get3D(speedY, x, y, z) = abs(velocity.y);
	get3D(speedZ, x, y, z) = abs(velocity.z);
}




MAC_Grid_3D::MAC_Grid_3D(int X, int Y, int Z, float cellPhysicalSize_) :
	sizeX(X), sizeY(Y), sizeZ(Z), cellCount((X + 1)* (Y + 1)* (Z + 1)), cellPhysicalSize(cellPhysicalSize_),
	physicalSizeX(X* cellPhysicalSize), physicalSizeY(Y* cellPhysicalSize)
{

	cudaGridSize = dim3(divUp(sizeX+1, cudaBlockSize.x), divUp(sizeY+1, cudaBlockSize.y), divUp(sizeZ+1, cudaBlockSize.z));

	std::cout << "cudaGridSize " << cudaGridSize.x << " " << cudaGridSize.y << " " << cudaGridSize.z << std::endl;


	volumes.content = createField3D<int>(sizeX + 1, sizeY + 1, sizeZ + 1, cudaGridSize,cudaBlockSize,CONTENT_AIR,false);
	volumes.pressure = createField3D<float>(sizeX , sizeY, sizeZ , cudaGridSize, cudaBlockSize, 0.f,false);
	volumes.fluidIndex = createField3D<int>(sizeX , sizeY , sizeZ , cudaGridSize, cudaBlockSize, 0, false);
	volumes.divergence = createField3D<float>(sizeX, sizeY, sizeZ, cudaGridSize, cudaBlockSize, 0.f, false);
	volumes.particleCount = createField3D<int>(sizeX, sizeY, sizeZ, cudaGridSize, cudaBlockSize, 0, false);

	volumes.velocityAccumWeight = createField3D<float4>(sizeX, sizeY, sizeZ, cudaGridSize, cudaBlockSize, make_float4(0,0,0,0), false);
	volumes.hasVelocity = createField3D<int4>(sizeX, sizeY, sizeZ, cudaGridSize, cudaBlockSize, make_int4(0, 0, 0, 0), false);


	volumes.velocity = createField3D<float4>(sizeX+1, sizeY+1, sizeZ+1, cudaGridSize, cudaBlockSize, make_float4(0, 0, 0, 0), true);

	volumes.newVelocity = createField3D<float4>(sizeX + 1, sizeY + 1, sizeZ + 1, cudaGridSize, cudaBlockSize, make_float4(0, 0, 0, 0), true);


	volumes.volumeFractions = createField3D<float4>(sizeX  , sizeY , sizeZ , cudaGridSize, cudaBlockSize, make_float4(0, 0, 0, 0), false);
	volumes.newVolumeFractions = createField3D<float4>(sizeX , sizeY , sizeZ , cudaGridSize, cudaBlockSize, make_float4(0, 0, 0, 0), false);



	updateFluidCount();
}




void MAC_Grid_3D::updateFluidCount() {
	
	unsigned int* fluidCountDevice;
	HANDLE_ERROR(cudaMalloc(&fluidCountDevice, sizeof(*fluidCountDevice)));
	HANDLE_ERROR(cudaMemset(fluidCountDevice, 0,sizeof(*fluidCountDevice)));

	setFluidIndex <<<cudaGridSize,cudaBlockSize>>> (volumes, cellCount, fluidCountDevice,sizeX,sizeY,sizeZ);
	CHECK_CUDA_ERROR("set fluid index");

	HANDLE_ERROR(cudaMemcpy(&fluidCount, fluidCountDevice, sizeof(fluidCount), cudaMemcpyDeviceToHost));

	std::cout << "current fluid cell count: " << fluidCount << std::endl;


	HANDLE_ERROR(cudaFree(fluidCountDevice));

}



float MAC_Grid_3D::getMaxSpeed() {
	float* speedX;
	float* speedY;
	float* speedZ;

	HANDLE_ERROR(cudaMalloc(&speedX, cellCount * sizeof(*speedX)));
	HANDLE_ERROR(cudaMalloc(&speedY, cellCount * sizeof(*speedY)));
	HANDLE_ERROR(cudaMalloc(&speedZ, cellCount * sizeof(*speedZ)));

	writeSpeed <<<cudaGridSize, cudaBlockSize >> > (volumes, cellCount, speedX,speedY,speedZ,sizeX,sizeY,sizeZ);
	CHECK_CUDA_ERROR("write speed");


	float maxX = thrust::reduce(thrust::device, speedX, speedX + cellCount, 0, thrust::maximum<float>());
	float maxY = thrust::reduce(thrust::device, speedY, speedY + cellCount, 0, thrust::maximum<float>());
	float maxZ = thrust::reduce(thrust::device, speedZ, speedZ + cellCount, 0, thrust::maximum<float>());

	float maxSpeed = max(max(maxX, maxY), maxZ) * sqrt(3);

	HANDLE_ERROR(cudaFree(speedX));
	HANDLE_ERROR(cudaFree(speedY));
	HANDLE_ERROR(cudaFree(speedZ));

	return maxSpeed;
}

MAC_Grid_3D::~MAC_Grid_3D() {
	releaseField3D(volumes. content);
	releaseField3D(volumes. pressure);
	releaseField3D(volumes. fluidIndex);
	releaseField3D(volumes. divergence);
	releaseField3D(volumes. particleCount);

	releaseField3D(volumes. velocityAccumWeight);
	releaseField3D(volumes. hasVelocity);

	releaseField3D(volumes. velocity);
	releaseField3D(volumes. newVelocity);

	releaseField3D(volumes. volumeFractions);
	releaseField3D(volumes. newVolumeFractions);
}