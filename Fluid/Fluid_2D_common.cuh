#pragma once

#include <thrust/functional.h>
#include <thrust/reduce.h>

#include "Fluid_2D_kernels.cuh"

template<typename Particle>
void performSpatialHashing(int* particleHashes, Particle* particles, int particleCount, float cellPhysicalSize, float sizeX, float sizeY, int numBlocksParticle, int numThreadsParticle, int* cellStart, int* cellEnd, int cellCount) {
	calcHashImpl<Particle> << < numBlocksParticle, numThreadsParticle >> > (particleHashes, particles, particleCount, cellPhysicalSize, sizeX, sizeY);
	CHECK_CUDA_ERROR("calc hash");

	thrust::sort_by_key(thrust::device, particleHashes, particleHashes + particleCount, particles);

	HANDLE_ERROR(cudaMemset(cellStart, 255, cellCount * sizeof(*cellStart)));
	HANDLE_ERROR(cudaMemset(cellEnd, 255, cellCount * sizeof(*cellEnd)));
	findCellStartEndImpl << < numBlocksParticle, numThreadsParticle >> > (particleHashes, cellStart, cellEnd, particleCount);
	CHECK_CUDA_ERROR("find cell start end");

}

void applyForces(float timeStep, MAC_Grid_2D& grid, float gravitationalAcceleration) {
	applyForcesImpl <<< grid.numBlocksCell, grid.numThreadsCell >> >
		(grid.cells, grid.sizeX, grid.sizeY, timeStep, gravitationalAcceleration);
	cudaDeviceSynchronize();
	CHECK_CUDA_ERROR("apply forces");

}

void fixBoundary( MAC_Grid_2D& grid) {
	int sizeX = grid.sizeX;
	int sizeY = grid.sizeY;

	int numThreads, numBlocks;

	numThreads = min(1024, sizeY);
	numBlocks = divUp(sizeY, numThreads);
	fixBoundaryX <<< numBlocks, numThreads >>> (grid.cells, sizeX, sizeY);
	cudaDeviceSynchronize();
	CHECK_CUDA_ERROR("fix boundary x");

	numThreads = min(1024, sizeX);
	numBlocks = divUp(sizeX, numThreads);
	fixBoundaryY <<< numBlocks, numThreads >>> (grid.cells, sizeX, sizeY);
	CHECK_CUDA_ERROR("fix boundary y");

}

void solvePressureJacobi(float timeStep,MAC_Grid_2D& grid, int iterations) {
	computeDivergenceImpl <<< grid.numBlocksCell, grid.numThreadsCell >>> 
		(grid.cells, grid.sizeX, grid.sizeY, grid.cellPhysicalSize);
	resetPressureImpl <<< grid.numBlocksCell, grid.numThreadsCell >>> (grid.cells, grid.sizeX, grid.sizeY);


	float dt_div_rho_div_dx = 1;

	for (int i = 0; i < iterations; ++i) {
		jacobiImpl <<< grid.numBlocksCell, grid.numThreadsCell >>> 
			(grid.cells, grid.sizeX, grid.sizeY, dt_div_rho_div_dx, grid.cellPhysicalSize);
	}

}

void solvePressure(float timeStep, MAC_Grid_2D& grid) {

	int sizeX = grid.sizeX;
	int sizeY = grid.sizeY;
	int numBlocksCell = grid.numBlocksCell;
	int numThreadsCell = grid.numThreadsCell;

	PressureEquation2D* equations = new PressureEquation2D[grid.fluidCount];
	int nnz = 0;
	bool hasNonZeroRHS = false;
	float dt_div_rho_div_dx = 1;


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

void updateVelocityWithPressure(float timeStep, MAC_Grid_2D& grid) {
	float dt_div_rho_div_dx = 1;
	updateVelocityWithPressureImpl <<< grid.numBlocksCell, grid.numThreadsCell >>> (grid.cells, grid.sizeX, grid.sizeY, dt_div_rho_div_dx);
	cudaDeviceSynchronize();
	CHECK_CUDA_ERROR("update velocity with pressure");
}


void extrapolateVelocity(float timeStep, MAC_Grid_2D& grid) {

	//used to decide how far to extrapolate
	float maxSpeed = grid.getMaxSpeed();

	float maxDist = (maxSpeed * timeStep + 1) / grid.cellPhysicalSize;
	//maxDist=4;
	//std::cout<<"maxDist "<<maxDist<<std::endl;

	for (int distance = 0; distance < maxDist; ++distance) {
		extrapolateVelocityByOne <<< grid.numBlocksCell, grid.numThreadsCell >> > (grid.cells, grid.sizeX, grid.sizeY);
		cudaDeviceSynchronize();
		CHECK_CUDA_ERROR("extrapolate vel");
	}
}


template<typename Particle>
void drawParticles(unsigned char* imageGPU,float containerSizeX,float containerSizeY,
	Particle* particles,int particleCount,int numBlocksParticle,int numThreadsParticle,
	int imageSizeX, int imageSizeY) {

	int imageMemorySize = imageSizeX * imageSizeY * 4;
	bool drawParticlesInsteadOfGrid = true;
	HANDLE_ERROR(cudaMemset(imageGPU, 255, imageMemorySize));


	drawParticleImpl << <numBlocksParticle, numThreadsParticle >> > ( containerSizeX,  containerSizeY, particles, particleCount, imageGPU, imageSizeX, imageSizeY);
	cudaDeviceSynchronize();
	CHECK_CUDA_ERROR("update tex");

}

void drawGrid(MAC_Grid_2D& grid,unsigned char* imageGPU) {
	drawCellImpl <<< grid.numBlocksCell, grid.numThreadsCell >> > (grid.cells, grid.sizeX, grid.sizeY, imageGPU);
	cudaDeviceSynchronize();
	CHECK_CUDA_ERROR("update tex");
}

