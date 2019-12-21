
#ifndef AQUARIUS_SPD_SOLVER_H
#define AQUARIUS_SPD_SOLVER_H

#include "../GpuCommons.h"
#include <iostream>


struct SparseMatrixCSR{
    int rows;
    int cols;
    double* val;
    int* csrRowPtr;
    int* csrColInd;
    cusparseMatDescr_t descr;
    int nnz;
    SparseMatrixCSR(int rows_,int cols_,double* val_,int*csrRowPtr_,int*csrColInd_,cusparseMatDescr_t descr_,int nnz_):
    rows(rows_),cols(cols_),val(val_),csrRowPtr(csrRowPtr_),csrColInd(csrColInd_),descr(descr_),nnz(nnz_){

    }
    void free(){
        HANDLE_ERROR(cudaFree(val));
        HANDLE_ERROR(cudaFree(csrRowPtr));
        HANDLE_ERROR(cudaFree(csrColInd));
    }
};

inline SparseMatrixCSR createSparse(double* valA,int rows,int cols){

    const cusparseHandle_t cusparseHandle = CudaHandlesKeeper::instance().cusparseHandle;
    //create device array and copy host to it
    double *d_A_dense;
    HANDLE_ERROR(cudaMalloc(&d_A_dense, rows * cols * sizeof(*d_A_dense)));
    HANDLE_ERROR(cudaMemcpy(d_A_dense, valA, rows * cols * sizeof(*d_A_dense), cudaMemcpyHostToDevice));

    // --- Descriptor for sparse matrix A
    cusparseMatDescr_t descrA;      HANDLE_ERROR(cusparseCreateMatDescr(&descrA));
    cusparseSetMatType      (descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase (descrA, CUSPARSE_INDEX_BASE_ZERO);

    int nnz = 0;                                // --- Number of nonzero elements in dense matrix
    const int lda = rows;                      // --- Leading dimension of dense matrix
    // --- Device side number of nonzero elements per row
    int *d_nnzPerVector;
    HANDLE_ERROR(cudaMalloc(&d_nnzPerVector, rows * sizeof(*d_nnzPerVector)));
    HANDLE_ERROR(cusparseDnnz(cusparseHandle, CUSPARSE_DIRECTION_ROW, rows, cols, descrA, d_A_dense, lda, d_nnzPerVector, &nnz));
    // --- Host side number of nonzero elements per row
    int *h_nnzPerVector = (int *)malloc(rows * sizeof(*h_nnzPerVector));
    HANDLE_ERROR(cudaMemcpy(h_nnzPerVector, d_nnzPerVector, rows * sizeof(*h_nnzPerVector), cudaMemcpyDeviceToHost));

    printf("Number of nonzero elements in dense matrix = %i\n\n", nnz);
    for (int i = 0; i < rows; ++i) printf("Number of nonzero elements in row %i = %i \n", i, h_nnzPerVector[i]);
    printf("\n");

    // --- Device side sparse matrix
    double *d_A;            HANDLE_ERROR(cudaMalloc(&d_A, nnz * sizeof(*d_A)));
    int *d_A_RowIndices;    HANDLE_ERROR(cudaMalloc(&d_A_RowIndices, (rows + 1) * sizeof(*d_A_RowIndices)));
    int *d_A_ColIndices;    HANDLE_ERROR(cudaMalloc(&d_A_ColIndices, nnz * sizeof(*d_A_ColIndices)));

    HANDLE_ERROR(cusparseDdense2csr(cusparseHandle, rows, cols, descrA, d_A_dense, lda, d_nnzPerVector, d_A, d_A_RowIndices, d_A_ColIndices));

    // --- Host side sparse matrix
    double *h_A = (double *)malloc(nnz * sizeof(*h_A));
    int *h_A_RowIndices = (int *)malloc((rows + 1) * sizeof(*h_A_RowIndices));
    int *h_A_ColIndices = (int *)malloc(nnz * sizeof(*h_A_ColIndices));
    HANDLE_ERROR(cudaMemcpy(h_A, d_A, nnz*sizeof(*h_A), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(h_A_RowIndices, d_A_RowIndices, (rows + 1) * sizeof(*h_A_RowIndices), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(h_A_ColIndices, d_A_ColIndices, nnz * sizeof(*h_A_ColIndices), cudaMemcpyDeviceToHost));

    for (int i = 0; i < nnz; ++i) printf("A[%i] = %.2f ", i, h_A[i]); printf("\n");

    for (int i = 0; i < (rows + 1); ++i) printf("h_A_RowIndices[%i] = %i \n", i, h_A_RowIndices[i]); printf("\n");

    for (int i = 0; i < nnz; ++i) printf("h_A_ColIndices[%i] = %i \n", i, h_A_ColIndices[i]);

    SparseMatrixCSR mat(rows,cols,d_A,d_A_RowIndices,d_A_ColIndices,descrA,nnz);

    // released used memory
    HANDLE_ERROR(cudaFree(d_A_dense));
    free(h_A);
    free(h_A_RowIndices);
    free(h_A_ColIndices);

    return mat;

}


struct DoubleToSingleCast : public thrust::unary_function<double, float>
{
	__host__ __device__
		float operator()(double x) { return x; }
};

struct SingleToDoubleCast : public thrust::unary_function<float,double>
{
	__host__ __device__
		double operator()(float x) { return x; }
};


inline double* solveSPD5(SparseMatrixCSR A, SparseMatrixCSR R, double* f_dense_host, int n) {

	AMGX_resources_handle amgxResource = CudaHandlesKeeper::instance().amgxResource;
	AMGX_config_handle SPD_solver_config = CudaHandlesKeeper::instance().SPD_solver_config;
	AMGX_solver_handle SPD_solver = CudaHandlesKeeper::instance().SPD_solver_single;

	double* fDouble;
	HANDLE_ERROR(cudaMalloc(&fDouble, n * sizeof(*fDouble)));
	HANDLE_ERROR(cudaMemcpy(fDouble, f_dense_host, n * sizeof(*fDouble), cudaMemcpyHostToDevice));

	float* xSingle;
	HANDLE_ERROR(cudaMalloc(&xSingle, n * sizeof(*xSingle)));

	double* xDouble;
	HANDLE_ERROR(cudaMalloc(&xDouble, n * sizeof(*xDouble)));

	DoubleToSingleCast doubleToSingle;

	float* AValSingle;
	HANDLE_ERROR(cudaMalloc(&AValSingle, A.nnz * sizeof(*AValSingle)));
	thrust::transform(thrust::device, A.val, A.val + A.nnz, AValSingle, doubleToSingle);

	float* fSingle;
	HANDLE_ERROR(cudaMalloc(&fSingle, n * sizeof(*fSingle)));
	thrust::transform(thrust::device, fDouble, fDouble+n, fSingle, doubleToSingle);


	AMGX_matrix_handle matrix;
	AMGX_vector_handle rhs;
	AMGX_vector_handle soln;


	AMGX_matrix_create(&matrix, amgxResource, AMGX_mode_dFFI);
	AMGX_vector_create(&rhs, amgxResource, AMGX_mode_dFFI);
	AMGX_vector_create(&soln, amgxResource, AMGX_mode_dFFI);

	AMGX_matrix_upload_all(matrix, n, A.nnz, 1, 1, A.csrRowPtr, A.csrColInd, AValSingle, nullptr);
	AMGX_vector_upload(rhs, n, 1, fSingle);
	AMGX_vector_set_zero(soln, n, 1);

	AMGX_solver_setup(SPD_solver, matrix);
	AMGX_solver_solve_with_0_initial_guess(SPD_solver, rhs, soln);

	AMGX_vector_download(soln, xSingle);


	AMGX_matrix_destroy(matrix);
	AMGX_vector_destroy(rhs);
	AMGX_vector_destroy(soln);

	//AMGX_config_destroy(solver_config);
	//AMGX_solver_destroy(solver);

	SingleToDoubleCast singleToDouble;
	thrust::transform(thrust::device, xSingle, xSingle+n, xDouble,singleToDouble);


	HANDLE_ERROR(cudaFree(fDouble));
	HANDLE_ERROR(cudaFree(fSingle));
	HANDLE_ERROR(cudaFree(AValSingle));
	HANDLE_ERROR(cudaFree(xSingle));



	return xDouble;
}



inline double* solveSPD4(SparseMatrixCSR A, SparseMatrixCSR R, double* f_dense_host,int n){

    AMGX_resources_handle amgxResource = CudaHandlesKeeper::instance().amgxResource;
    AMGX_config_handle SPD_solver_config = CudaHandlesKeeper::instance().SPD_solver_config;
    AMGX_solver_handle SPD_solver = CudaHandlesKeeper::instance().SPD_solver_double;

    double* f;
    HANDLE_ERROR(cudaMalloc(&f, n * sizeof(*f)));
    HANDLE_ERROR(cudaMemcpy(f, f_dense_host, n * sizeof(*f), cudaMemcpyHostToDevice));

    double* x;
    HANDLE_ERROR(cudaMalloc(&x, n * sizeof(*x)));

    AMGX_matrix_handle matrix;
    AMGX_vector_handle rhs;
    AMGX_vector_handle soln;


    AMGX_matrix_create(&matrix, amgxResource, AMGX_mode_dDDI);
    AMGX_vector_create(&rhs, amgxResource, AMGX_mode_dDDI);
    AMGX_vector_create(&soln, amgxResource, AMGX_mode_dDDI);

    AMGX_matrix_upload_all(matrix, n,A.nnz,1,1,A.csrRowPtr,A.csrColInd,A.val, nullptr);
    AMGX_vector_upload(rhs, n,1,f);
    AMGX_vector_set_zero(soln, n,1);

    AMGX_solver_setup(SPD_solver, matrix);
    AMGX_solver_solve_with_0_initial_guess(SPD_solver, rhs, soln);

    AMGX_vector_download(soln, x);


    AMGX_matrix_destroy(matrix);
    AMGX_vector_destroy(rhs);
    AMGX_vector_destroy(soln);

    //AMGX_config_destroy(solver_config);
    //AMGX_solver_destroy(solver);

    HANDLE_ERROR(cudaFree(f));

    return x;
}


//cusparse plain CG
inline double* solveSPD3(SparseMatrixCSR A, SparseMatrixCSR R, double* f_dense_host,int n){

    const cusparseHandle_t cusparseHandle = CudaHandlesKeeper::instance().cusparseHandle;
    const cublasHandle_t cublasHandle = CudaHandlesKeeper::instance().cublasHandle;
    /***** CG Code *****/
/* ASSUMPTIONS:
   1. The cuSPARSE and cuBLAS libraries have been initialized.
   2. The appropriate memory has been allocated and set to zero.
   3. The matrix A (valA, csrRowPtrA, csrColIndA) and the incomplete-
      Cholesky upper triangular factor R (valR, csrRowPtrR, csrColIndR)
      have been computed and are present in the device (GPU) memory. */

//create the info and analyse the lower and upper triangular factors

    double* valA = A.val;
    int* csrRowPtrA = A.csrRowPtr;
    int* csrColIndA = A.csrColInd;
    cusparseMatDescr_t descrA = A.descr;


    double* f;
    HANDLE_ERROR(cudaMalloc(&f, n * sizeof(*f)));
    HANDLE_ERROR(cudaMemcpy(f, f_dense_host, n * sizeof(*f), cudaMemcpyHostToDevice));

    double* x;
    HANDLE_ERROR(cudaMalloc(&x, n * sizeof(*x)));
    HANDLE_ERROR(cudaMemset(x,0, n * sizeof(*x)));

    double* r;
    HANDLE_ERROR(cudaMalloc(&r, n * sizeof(*r)));

    double* z;
    HANDLE_ERROR(cudaMalloc(&z, n * sizeof(*z)));

    double* t;
    HANDLE_ERROR(cudaMalloc(&t, n * sizeof(*t)));

    double* p;
    HANDLE_ERROR(cudaMalloc(&p, n * sizeof(*p)));

    double* q;
    HANDLE_ERROR(cudaMalloc(&q, n * sizeof(*q)));

    cusparseHandle_t handle = cusparseHandle;

    double nrmr0 = 0.0;
    double nrmr = 0.0;
    double rhop = 0.0;
    double rho = 0.0;
    double alpha = 0.0;
    double beta = 0.0;
    double temp = 0.0;
    double minus_alpha = 0.0;

    int maxit = 2000;

    double tol = 1e-6;

    const double d_1 = -1;
    const double d0 = 0;
    const double d1 = 1;



//1: compute initial residual r = f -  A x0 (using initial guess in x)
    HANDLE_ERROR(cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, A.nnz,&d1,
                                descrA, valA, csrRowPtrA, csrColIndA, x, &d0, r));
    HANDLE_ERROR(cublasDscal(cublasHandle,n,&d_1, r, 1));
    HANDLE_ERROR(cublasDaxpy(cublasHandle,n, &d1, f, 1, r, 1));
    HANDLE_ERROR(cublasDnrm2(cublasHandle,n, r, 1,&nrmr0));

//2: repeat until convergence (based on max. it. and relative residual)
    int i;
    for (i=0; i<maxit; i++){
        //3: Solve M z = r (sparse lower and upper triangular solves)
        HANDLE_ERROR(cublasDcopy(cublasHandle,n, r, 1, z, 1));

        //4: \rho = r^{T} z
        rhop= rho;
        HANDLE_ERROR(cublasDdot(cublasHandle,n, r, 1, z, 1,&rho));
        if (i == 0){
            //6: p = z
            HANDLE_ERROR(cublasDcopy(cublasHandle,n, z, 1, p, 1));
        }
        else{
            //8: \beta = rho_{i} / \rho_{i-1}
            beta= rho/rhop;
            //9: p = z + \beta p
            HANDLE_ERROR(cublasDaxpy(cublasHandle,n, &beta, p, 1, z, 1));
            HANDLE_ERROR(cublasDcopy(cublasHandle,n, z, 1, p, 1));
        }

        //11: Compute q = A p (sparse matrix-vector multiplication)
        HANDLE_ERROR(cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, A.nnz,&d1,
                                    descrA, valA, csrRowPtrA, csrColIndA, p, &d0, q));

        //12: \alpha = \rho_{i} / (p^{T} q)
        HANDLE_ERROR(cublasDdot(cublasHandle,n, p, 1, q, 1,&temp));
        alpha= rho/temp;
        //13: x = x + \alpha p
        HANDLE_ERROR(cublasDaxpy(cublasHandle,n, &alpha, p, 1, x, 1));
        //14: r = r - \alpha q
        minus_alpha = -alpha;
        HANDLE_ERROR(cublasDaxpy(cublasHandle,n,&minus_alpha, q, 1, r, 1));

        //check for convergence
        HANDLE_ERROR(cublasDdot(cublasHandle,n, r, 1, r, 1,&nrmr));
        if (sqrt(nrmr)/nrmr0 < tol){
            break;
        }
        //std::cout<<"iterations: "<<i<<"          "<<"error: "<<nrmr<<std::endl;

    }

    std::cout<<"iterations: "<<i<<"          "<<"error: "<<nrmr<<std::endl;

    //release memory

    HANDLE_ERROR(cudaFree(f));
    HANDLE_ERROR(cudaFree(r));
    HANDLE_ERROR(cudaFree(z));
    HANDLE_ERROR(cudaFree(t));
    HANDLE_ERROR(cudaFree(p));
    HANDLE_ERROR(cudaFree(q));

    return x;

}

//using cusolver (IC0PCG)
inline double* solveSPD2(SparseMatrixCSR A, SparseMatrixCSR R, double* f_dense_host,int n){


    const cusparseHandle_t cusparseHandle = CudaHandlesKeeper::instance().cusparseHandle;
    const cublasHandle_t cublasHandle = CudaHandlesKeeper::instance().cublasHandle;
    cusolverSpHandle_t cusolverSpHandle = CudaHandlesKeeper::instance().cusolverSpHandle;

    double* valA = A.val;
    int* csrRowPtrA = A.csrRowPtr;
    int* csrColIndA = A.csrColInd;
    cusparseMatDescr_t descrA = A.descr;

    double* f;
    HANDLE_ERROR(cudaMalloc(&f, n * sizeof(*f)));
    HANDLE_ERROR(cudaMemcpy(f, f_dense_host, n * sizeof(*f), cudaMemcpyHostToDevice));

    double* x;
    HANDLE_ERROR(cudaMalloc(&x, n * sizeof(*x)));
    HANDLE_ERROR(cudaMemset(x,0, n * sizeof(*x)));

    int singularity = 0;

    cusolverSpDcsrlsvchol(cusolverSpHandle,n,A.nnz,descrA,valA,csrRowPtrA,csrColIndA,f,1e-6,0,x,&singularity);

	if (singularity != -1) {
		std::cout << "When using cusolver, detected that the matrix isn't SPD!! " << std::endl;
	}

	HANDLE_ERROR(cudaFree(f));
    return x;
}

inline double* solveSPD(SparseMatrixCSR A, SparseMatrixCSR R, double* f_dense_host,int n){


    const cusparseHandle_t cusparseHandle = CudaHandlesKeeper::instance().cusparseHandle;
    const cublasHandle_t cublasHandle = CudaHandlesKeeper::instance().cublasHandle;
    /***** CG Code *****/
/* ASSUMPTIONS:
   1. The cuSPARSE and cuBLAS libraries have been initialized.
   2. The appropriate memory has been allocated and set to zero.
   3. The matrix A (valA, csrRowPtrA, csrColIndA) and the incomplete-
      Cholesky upper triangular factor R (valR, csrRowPtrR, csrColIndR)
      have been computed and are present in the device (GPU) memory. */

//create the info and analyse the lower and upper triangular factors

    double* valA = A.val;
    int* csrRowPtrA = A.csrRowPtr;
    int* csrColIndA = A.csrColInd;
    cusparseMatDescr_t descrA = A.descr;

    double* valR = R.val;
    int* csrRowPtrR = R.csrRowPtr;
    int* csrColIndR = R.csrColInd;
    cusparseMatDescr_t descrR = R.descr;

    double* f;
    HANDLE_ERROR(cudaMalloc(&f, n * sizeof(*f)));
    HANDLE_ERROR(cudaMemcpy(f, f_dense_host, n * sizeof(*f), cudaMemcpyHostToDevice));

    double* x;
    HANDLE_ERROR(cudaMalloc(&x, n * sizeof(*x)));
    HANDLE_ERROR(cudaMemset(x,0, n * sizeof(*x)));

    double* r;
    HANDLE_ERROR(cudaMalloc(&r, n * sizeof(*r)));

    double* z;
    HANDLE_ERROR(cudaMalloc(&z, n * sizeof(*z)));

    double* t;
    HANDLE_ERROR(cudaMalloc(&t, n * sizeof(*t)));

    double* p;
    HANDLE_ERROR(cudaMalloc(&p, n * sizeof(*p)));

    double* q;
    HANDLE_ERROR(cudaMalloc(&q, n * sizeof(*q)));

    cusparseHandle_t handle = cusparseHandle;

    cusparseSolveAnalysisInfo_t inforRt;
    cusparseSolveAnalysisInfo_t inforR;

    double nrmr0 = 0.0;
    double nrmr = 0.0;
    double rhop = 0.0;
    double rho = 0.0;
    double alpha = 0.0;
    double beta = 0.0;
    double temp = 0.0;
    double minus_alpha = 0.0;

    int maxit = 2000;

    double tol = 1e-6;

    const double d_1 = -1;
    const double d0 = 0;
    const double d1 = 1;


    HANDLE_ERROR(cusparseCreateSolveAnalysisInfo(&inforRt));
    HANDLE_ERROR(cusparseCreateSolveAnalysisInfo(&inforR));
    HANDLE_ERROR(cusparseDcsrsv_analysis(handle,CUSPARSE_OPERATION_TRANSPOSE,n,R.nnz,
            descrR, valR, csrRowPtrR, csrColIndR, inforRt));
    HANDLE_ERROR(cusparseDcsrsv_analysis(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,n,R.nnz,
                            descrR, valR, csrRowPtrR, csrColIndR, inforR));

//1: compute initial residual r = f -  A x0 (using initial guess in x)
    HANDLE_ERROR(cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, A.nnz,&d1,
                   descrA, valA, csrRowPtrA, csrColIndA, x, &d0, r));
    HANDLE_ERROR(cublasDscal(cublasHandle,n,&d_1, r, 1));
    HANDLE_ERROR(cublasDaxpy(cublasHandle,n, &d1, f, 1, r, 1));
    HANDLE_ERROR(cublasDnrm2(cublasHandle,n, r, 1,&nrmr0));

//2: repeat until convergence (based on max. it. and relative residual)
    int i;
    for (i=0; i<maxit; i++){
        //3: Solve M z = r (sparse lower and upper triangular solves)
        HANDLE_ERROR(cusparseDcsrsv_solve(handle, CUSPARSE_OPERATION_TRANSPOSE,
                             n, &d1, descrR, valR, csrRowPtrR, csrColIndR,
                             inforRt, r, t));
        HANDLE_ERROR(cusparseDcsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                             n, &d1, descrR, valR, csrRowPtrR, csrColIndR,
                             inforR, t, z));

        //4: \rho = r^{T} z
        rhop= rho;
        HANDLE_ERROR(cublasDdot(cublasHandle,n, r, 1, z, 1,&rho));
        if (i == 0){
            //6: p = z
            HANDLE_ERROR(cublasDcopy(cublasHandle,n, z, 1, p, 1));
        }
        else{
            //8: \beta = rho_{i} / \rho_{i-1}
            beta= rho/rhop;
            //9: p = z + \beta p
            HANDLE_ERROR(cublasDaxpy(cublasHandle,n, &beta, p, 1, z, 1));
            HANDLE_ERROR(cublasDcopy(cublasHandle,n, z, 1, p, 1));
        }

        //11: Compute q = A p (sparse matrix-vector multiplication)
        HANDLE_ERROR(cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, A.nnz,&d1,
                       descrA, valA, csrRowPtrA, csrColIndA, p, &d0, q));

        //12: \alpha = \rho_{i} / (p^{T} q)
        HANDLE_ERROR(cublasDdot(cublasHandle,n, p, 1, q, 1,&temp));
        alpha= rho/temp;
        //13: x = x + \alpha p
        HANDLE_ERROR(cublasDaxpy(cublasHandle,n, &alpha, p, 1, x, 1));
        //14: r = r - \alpha q
        minus_alpha = -alpha;
        HANDLE_ERROR(cublasDaxpy(cublasHandle,n,&minus_alpha, q, 1, r, 1));

        //check for convergence
        HANDLE_ERROR(cublasDnrm2(cublasHandle,n, r, 1,&nrmr));
        if (nrmr/nrmr0 < tol){
            break;
        }
        //std::cout<<"iterations: "<<i<<"          "<<"error: "<<nrmr<<std::endl;

    }

//destroy the analysis info (for lower and upper triangular factors)
    HANDLE_ERROR(cusparseDestroySolveAnalysisInfo(inforRt));
    HANDLE_ERROR(cusparseDestroySolveAnalysisInfo(inforR));

    std::cout<<"iterations: "<<i<<"          "<<"error: "<<nrmr<<std::endl;

    //release memory

    HANDLE_ERROR(cudaFree(f));
    HANDLE_ERROR(cudaFree(r));
    HANDLE_ERROR(cudaFree(z));
    HANDLE_ERROR(cudaFree(t));
    HANDLE_ERROR(cudaFree(p));
    HANDLE_ERROR(cudaFree(q));

    return x;

}

#endif //AQUARIUS_SPD_SOLVER_H
