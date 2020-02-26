#pragma once

#include "../Common/GpuCommons.h"
__device__ __host__
struct PressureEquation2D {
	int termsIndex[5];
	float termsCoeff[5];
	unsigned char termCount = 0;
	float RHS;
	int x;
	int y;
};

__device__ __host__
struct PressureEquation3D {
	int termsIndex[7];
	float termsCoeff[7];
	unsigned char termCount = 0;
	float RHS;
	int x;
	int y;
	int z;
};


void  solvePressure(float timeStep, MAC_Grid_3D& grid) {

	int sizeX = grid.sizeX;
	int sizeY = grid.sizeY;
	int sizeZ = grid.sizeZ;


	PressureEquation3D* equations = new PressureEquation3D[grid.fluidCount];
	int nnz = 0;
	bool hasNonZeroRHS = false;


	PressureEquation3D* equationsDevice;
	HANDLE_ERROR(cudaMalloc(&equationsDevice, grid.fluidCount * sizeof(PressureEquation3D)));

	bool* hasNonZeroRHS_Device;
	HANDLE_ERROR(cudaMalloc(&hasNonZeroRHS_Device, sizeof(*hasNonZeroRHS_Device)));
	HANDLE_ERROR(cudaMemset(hasNonZeroRHS_Device, 0, sizeof(*hasNonZeroRHS_Device)));

	constructPressureEquations << < grid.cudaGridSize, grid.cudaBlockSize >> >
		(grid.volumes, sizeX, sizeY, sizeZ, equationsDevice, hasNonZeroRHS_Device);
	cudaDeviceSynchronize();
	CHECK_CUDA_ERROR("construct eqns");


	HANDLE_ERROR(cudaMemcpy(equations, equationsDevice, grid.fluidCount * sizeof(PressureEquation3D),
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
		PressureEquation3D& thisEquation = equations[row];
		A_rowPtr_host[row] = i;

		for (int term = 0; term < thisEquation.termCount; ++term) {
			//if(thisEquation.termsIndex[term] > row) continue;
			A_host[i] = thisEquation.termsCoeff[term];
			A_colInd_host[i] = thisEquation.termsIndex[term];
			++i;
		}

	}

	for (int row = 0, i = 0; row < numVariables; ++row) {
		PressureEquation3D& thisEquation = equations[row];
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
	double* result_device = solveSPD2(A, R, f_host, numVariables);

	double* result_host = new double[numVariables];
	HANDLE_ERROR(cudaMemcpy(result_host, result_device, numVariables * sizeof(*result_host),
		cudaMemcpyDeviceToHost));


	setPressure << < grid.cudaGridSize, grid.cudaBlockSize >> > (grid.volumes, sizeX, sizeY, sizeZ, result_device);
	cudaDeviceSynchronize();
	CHECK_CUDA_ERROR("set pressure");



	A.free();
	R.free();
	free(f_host);
	HANDLE_ERROR(cudaFree(result_device));
	delete[](result_host);

	delete[] equations;

}




__device__ float getNeibourCoefficient(int x, int y, int z, float u, float& centerCoefficient, float& RHS, VolumeCollection volumes, int sizeX, int sizeY, int sizeZ) {

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

	if (volumes.content.readSurface<int>(x, y, z) != CONTENT_FLUID)
		return;

	PressureEquation3D thisEquation;
	float RHS = -volumes.divergence.readSurface<float>(x, y, z);

	float centerCoeff = 6;

	float4 thisNewVelocity = volumes.newVelocity.readSurface<float4>(x, y, z);
	float4 rightNewVelocity = volumes.newVelocity.readSurface<float4>(x + 1, y, z);
	float4 newVelocity = volumes.newVelocity.readSurface<float4>(x, y + 1, z);
	float4 frontNewVelocity = volumes.newVelocity.readSurface<float4>(x, y, z + 1);

	float leftCoeff = getNeibourCoefficient(x - 1, y, z, thisNewVelocity.x, centerCoeff, RHS, volumes, sizeX, sizeY, sizeZ);
	float rightCoeff = getNeibourCoefficient(x + 1, y, z, rightNewVelocity.x, centerCoeff, RHS, volumes, sizeX, sizeY, sizeZ);
	float downCoeff = getNeibourCoefficient(x, y - 1, z, thisNewVelocity.y, centerCoeff, RHS, volumes, sizeX, sizeY, sizeZ);
	float upCoeff = getNeibourCoefficient(x, y + 1, z, newVelocity.y, centerCoeff, RHS, volumes, sizeX, sizeY, sizeZ);
	float backCoeff = getNeibourCoefficient(x, y, z - 1, thisNewVelocity.z, centerCoeff, RHS, volumes, sizeX, sizeY, sizeZ);
	float frontCoeff = getNeibourCoefficient(x, y, z + 1, frontNewVelocity.z, centerCoeff, RHS, volumes, sizeX, sizeY, sizeZ);

	int nnz = 0;

	if (downCoeff) {
		int downIndex = volumes.fluidIndex.readSurface<int>(x, y - 1, z);
		thisEquation.termsIndex[thisEquation.termCount] = downIndex;
		thisEquation.termsCoeff[thisEquation.termCount] = downCoeff;
		++thisEquation.termCount;
		++nnz;
	}
	if (leftCoeff) {
		int leftIndex = volumes.fluidIndex.readSurface<int>(x - 1, y, z);
		thisEquation.termsIndex[thisEquation.termCount] = leftIndex;
		thisEquation.termsCoeff[thisEquation.termCount] = leftCoeff;
		++thisEquation.termCount;
		++nnz;
	}
	if (backCoeff) {
		int backIndex = volumes.fluidIndex.readSurface<int>(x, y, z - 1);
		thisEquation.termsIndex[thisEquation.termCount] = backIndex;
		thisEquation.termsCoeff[thisEquation.termCount] = backCoeff;
		++thisEquation.termCount;
		++nnz;
	}
	int thisIndex = volumes.fluidIndex.readSurface<int>(x, y, z);
	thisEquation.termsIndex[thisEquation.termCount] = thisIndex;
	thisEquation.termsCoeff[thisEquation.termCount] = centerCoeff;
	++thisEquation.termCount;
	if (rightCoeff) {
		int rightIndex = volumes.fluidIndex.readSurface<int>(x + 1, y, z);
		thisEquation.termsIndex[thisEquation.termCount] = rightIndex;
		thisEquation.termsCoeff[thisEquation.termCount] = rightCoeff;
		++thisEquation.termCount;
		++nnz;
	}
	if (upCoeff) {
		int upIndex = volumes.fluidIndex.readSurface<int>(x, y + 1, z);
		thisEquation.termsIndex[thisEquation.termCount] = upIndex;
		thisEquation.termsCoeff[thisEquation.termCount] = upCoeff;
		++thisEquation.termCount;
		++nnz;
	}
	if (frontCoeff) {
		int frontIndex = volumes.fluidIndex.readSurface<int>(x, y, z + 1);
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
