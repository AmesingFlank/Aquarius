#include <iostream>
#include <thrust/device_vector.h>
#include "CudaCommons.h"
#include <chrono>
#include <thread>


#include "PCG.h"
#include "Quad.h"
#include "Fluid_2D_SemiLagrange.h"

/*
__global__ void add( int a, int b, int *c ) {
    int result = a+b;
    *c = result;
    printf("haha\n");
}

void kernalTest(){
    int c;
    int *dev_c;
    HANDLE_ERROR( cudaMalloc( (void**)&dev_c, sizeof(int) ) );

    add<<<1,1>>>( 2, 7, dev_c );

    HANDLE_ERROR( cudaMemcpy( &c, dev_c, sizeof(int),
                              cudaMemcpyDeviceToHost ) );
    printf( "2 + 7 = %d\n", c );
    HANDLE_ERROR( cudaFree( dev_c ) );
}*/

__global__ void check_eq(double* a,double* b,int N){
    for (int i = 0; i < N ; ++i) {
        if(a[i]!=b[i]){
            printf("Not equal  %d \n",i);
        }
    }
    printf("finished checking equals\n");

}
__global__ void check_eq(int* a,int* b,int N){
    for (int i = 0; i < N ; ++i) {
        if(a[i]!=b[i]){
            printf("Not equal  %d \n",i);
        }
    }
    printf("finished checking equals\n");
}

void pcgTest(){
    double *h_A_dense = (double*)malloc(4*4*sizeof(*h_A_dense));
    // --- Column-major ordering
    h_A_dense[0] = 2.0; h_A_dense[4] = 3.0; h_A_dense[8]  = 1.0; h_A_dense[12] = 1.0;
    h_A_dense[1] = 3.0; h_A_dense[5] = 2.0; h_A_dense[9]  = 5.0; h_A_dense[13] = 1.0;
    h_A_dense[2] = 1.0; h_A_dense[6] = 5.0; h_A_dense[10] = 2.0; h_A_dense[14] = 7.0;
    h_A_dense[3] = 1.0; h_A_dense[7] = 1.0; h_A_dense[11] = 7.0; h_A_dense[15] = 4.0;

    double *h_R_dense = (double*)malloc(4*4*sizeof(*h_R_dense));
    // --- Column-major ordering
    h_R_dense[0] = 1.0; h_R_dense[4] = 0.1; h_R_dense[8]  = 0.0; h_R_dense[12] = 0.0;
    h_R_dense[1] = 0.0; h_R_dense[5] = 1.0; h_R_dense[9]  = 0.1; h_R_dense[13] = 0.0;
    h_R_dense[2] = 0.0; h_R_dense[6] = 0.0; h_R_dense[10] = 1.0; h_R_dense[14] = 0.1;
    h_R_dense[3] = 0.0; h_R_dense[7] = 0.0; h_R_dense[11] = 0.0; h_R_dense[15] = 1.0;

    double *h_f_dense = (double*)malloc(4*sizeof(*h_f_dense));
    h_f_dense[0] = 0.15f;
    h_f_dense[1] = 0.12f;
    h_f_dense[2]  = 0.15f;
    h_f_dense[3] = 0.1f;

    SparseMatrixCSR A = createSparse(h_A_dense,4,4);
    SparseMatrixCSR R = createSparse(h_R_dense,4,4);

    int n = 4;
    int nnz_R = 7;
    //nnz_R = n;
    double R_host[7];
    int R_rowPtr_host[5];
    int R_colInd_host[7];

    int i = 0;

    for (int row = 0; row<n;++row) {
        R_rowPtr_host[row] = i;
        for (int j = 0;j<n;++j){
            if(j==row){
                R_host[i] = 1;
                R_colInd_host[i]=j;
                ++i;
            }
            else if(j==row+1){
                R_host[i] = 0.1;
                R_colInd_host[i]=j;
                ++i;
            }
        }
    }

    std::cout<<i<<std::endl;
    std::cout<<nnz_R<<std::endl;


    R_rowPtr_host[n] = nnz_R;

    double *R_device;
    HANDLE_ERROR(cudaMalloc(&R_device, nnz_R * sizeof(*R_device)));
    HANDLE_ERROR(cudaMemcpy(R_device, R_host, nnz_R * sizeof(*R_device), cudaMemcpyHostToDevice));

    int *R_rowPtr_device;
    HANDLE_ERROR(cudaMalloc(&R_rowPtr_device, (n + 1) * sizeof(*R_rowPtr_device)));
    HANDLE_ERROR(cudaMemcpy(R_rowPtr_device, R_rowPtr_host, (n+1) * sizeof(*R_rowPtr_device), cudaMemcpyHostToDevice));

    int *R_colInd_device;
    HANDLE_ERROR(cudaMalloc(&R_colInd_device, nnz_R * sizeof(*R_colInd_device)));
    HANDLE_ERROR(cudaMemcpy(R_colInd_device, R_colInd_host, nnz_R * sizeof(*R_colInd_device), cudaMemcpyHostToDevice));

    cusparseMatDescr_t descrR;
    HANDLE_ERROR(cusparseCreateMatDescr(&descrR));
    cusparseSetMatType      (descrR, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase (descrR, CUSPARSE_INDEX_BASE_ZERO);

    SparseMatrixCSR R2(n,n,R_device,R_rowPtr_device,R_colInd_device,descrR,nnz_R);
    check_eq<<<1,1>>>(R2.csrColInd,R.csrColInd,nnz_R);
    check_eq<<<1,1>>>(R2.val,R.val,nnz_R);
    check_eq<<<1,1>>>(R2.csrColInd,R.csrColInd,nnz_R);
    if(R2.nnz!=R.nnz){
        std::cout<<"error nnz"<<std::endl;
    }
    if(R2.rows!=R.rows){
        std::cout<<"error r"<<std::endl;
    }
    if(R2.cols!=R.cols){
        std::cout<<"error c"<<std::endl;
    }

    double *x = solveSPD(A,R2,h_f_dense,4);


    A.free();
    R.free();


    double *h_x = (double *)malloc(4 * sizeof(*h_x));
    HANDLE_ERROR(cudaMemcpy(h_x, x, 4 * sizeof(*x), cudaMemcpyDeviceToHost));

    for (int i = 0; i < 4; ++i) {
        std::cout<<h_x[i]<<std::endl;
    }

}


cublasHandle_t cublasHandle;
cusparseHandle_t cusparseHandle;
cusolverSpHandle_t cusolverSpHandle;
AMGX_resources_handle amgxResource;

int main( void ) {
    initCuda();
    //kernalTest();
    //pcgTest(); return 0;

    //std::this_thread::sleep_for(std::chrono::milliseconds(10000));

    GLFWwindow* window = createWindowOpenGL(1024,512);

    Fluid_2D_SemiLagrange fluid;
    Quad quad;

    double framesSinceLast = 0;
    double lastSecond=glfwGetTime();
    double lastFrameTime = glfwGetTime();

    while(!glfwWindowShouldClose(window)){

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glClear(GL_COLOR_BUFFER_BIT);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
        glClearColor(0.25f,0.38f,0.5f,0.f);

        double currentTime = glfwGetTime();

        fluid.simulationStep(currentTime-lastFrameTime);
        fluid.updateTexture();
        quad.draw(fluid.texture);

        ++framesSinceLast;

        if(currentTime-lastSecond>=1){
            double FPS = (double)framesSinceLast/(currentTime-lastSecond);
            std::cout<<"FPS: "<<FPS<<std::endl;
            lastSecond = currentTime;
            framesSinceLast = 0;
        }

        lastFrameTime = currentTime;

        printGLError();
        glfwPollEvents();
        glfwSwapBuffers(window);
    }

    std::cout<<"finished everything"<<std::endl;

    return 0;
}
