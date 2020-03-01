#include "Fluid_3D_common.cuh"



void  applyGravity(float timeStep, MAC_Grid_3D& grid, float3 gravitationalAcceleration) {
	applyGravityImpl <<< grid.cudaGridSize, grid.cudaBlockSize >>>
		( grid.volumes,grid.sizeX, grid.sizeY, grid.sizeZ, timeStep, gravitationalAcceleration);
	cudaDeviceSynchronize();
	CHECK_CUDA_ERROR("apply forces");

}

void  fixBoundary(MAC_Grid_3D& grid) {
	int sizeX = grid.sizeX;
	int sizeY = grid.sizeY;
	int sizeZ = grid.sizeZ;

	int total, numThreads, numBlocks;

	total = sizeY * sizeZ;
	numThreads = min(1024, total);
	numBlocks = divUp(total, numThreads);
	fixBoundaryX << < numBlocks, numThreads >> > ( grid.volumes, sizeX, sizeY, sizeZ);
	cudaDeviceSynchronize();
	CHECK_CUDA_ERROR("fix boundary x");

	total = sizeX * sizeZ;
	numThreads = min(1024, total);
	numBlocks = divUp(total, numThreads);
	fixBoundaryY << < numBlocks, numThreads >> > ( grid.volumes, sizeX, sizeY, sizeZ);
	cudaDeviceSynchronize();
	CHECK_CUDA_ERROR("fix boundary y");

	total = sizeX * sizeY;
	numThreads = min(1024, total);
	numBlocks = divUp(total, numThreads);
	fixBoundaryZ << < numBlocks, numThreads >> > ( grid.volumes, sizeX, sizeY, sizeZ);
	cudaDeviceSynchronize();
	CHECK_CUDA_ERROR("fix boundary y");

}

void  computeDivergence(MAC_Grid_3D& grid, float restParticlesPerCell) {
	computeDivergenceImpl <<< grid.cudaGridSize, grid.cudaBlockSize >>>
		( grid.volumes, grid.sizeX, grid.sizeY, grid.sizeZ, grid.cellPhysicalSize, restParticlesPerCell);
}

void  solvePressureJacobi(float timeStep, MAC_Grid_3D& grid, int iterations) {

	for (int i = 0; i < iterations; ++i) {
		jacobiImpl  <<< grid.cudaGridSize, grid.cudaBlockSize >>>
			( grid.volumes, grid.sizeX, grid.sizeY, grid.sizeZ, grid.cellPhysicalSize);
	}

}


void  updateVelocityWithPressure(float timeStep, MAC_Grid_3D& grid) {
	updateVelocityWithPressureImpl <<< grid.cudaGridSize, grid.cudaBlockSize >>> ( grid.volumes, grid.sizeX, grid.sizeY, grid.sizeZ);
	cudaDeviceSynchronize();
	CHECK_CUDA_ERROR("update velocity with pressure");
}


void  extrapolateVelocity(float timeStep, MAC_Grid_3D& grid) {

	//used to decide how far to extrapolate
	//float maxSpeed = grid.getMaxSpeed();

	//float maxDist = ceil((maxSpeed * timeStep) / grid.cellPhysicalSize);
	//maxDist=4;
	//std::cout<<"maxDist "<<maxDist<<std::endl;

	int maxDistance = 3;

	for (int distance = 0; distance < maxDistance; ++distance) {
		//std::cout << "extrapolating" << std::endl;
		extrapolateVelocityByOne << < grid.cudaGridSize, grid.cudaBlockSize >> > ( grid.volumes, grid.sizeX, grid.sizeY, grid.sizeZ);
		cudaDeviceSynchronize();
		CHECK_CUDA_ERROR("extrapolate vel");
	}
}


void  solveDiffusionJacobi(float timeStep, MAC_Grid_3D& grid, int iterations, float D, float cellPhysicalSize) {
	float lambda = D * timeStep / (cellPhysicalSize * cellPhysicalSize);
	for (int i = 0; i < iterations; ++i) {
		diffusionJacobiImpl << < grid.cudaGridSize, grid.cudaBlockSize >> >
			(grid.volumes, grid.sizeX, grid.sizeY, grid.sizeZ, lambda);
	}
}