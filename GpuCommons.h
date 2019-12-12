//
// Created by AmesingFlank on 2019-04-17.
//

#ifndef AQUARIUS_GPUCOMMONS_H
#define AQUARIUS_GPUCOMMONS_H

#include <string>
#include <iostream>
#include <stdexcept>
#include <stdio.h>
#include <vcruntime_exception.h>
#include <helper_math.h>
#include <device_launch_parameters.h>

#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <cusolverSp.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <amgx/amgx_c.h>

#include <curand.h>
#include <curand_kernel.h>

#include "cuda_gl_interop.h"

#include <stdarg.h>

#include <thrust/functional.h>
#include <thrust/reduce.h>

inline const char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "unknown error";
}

inline const char* cusparseGetErrorString(cusparseStatus_t status)
{
    switch(status)
    {
        case CUSPARSE_STATUS_SUCCESS: return "CUSPARSE_STATUS_SUCCESS";
        case CUSPARSE_STATUS_NOT_INITIALIZED: return "CUSPARSE_STATUS_NOT_INITIALIZED";
        case CUSPARSE_STATUS_ALLOC_FAILED: return "CUSPARSE_STATUS_ALLOC_FAILED";
        case CUSPARSE_STATUS_INVALID_VALUE: return "CUSPARSE_STATUS_INVALID_VALUE";
        case CUSPARSE_STATUS_ARCH_MISMATCH: return "CUSPARSE_STATUS_ARCH_MISMATCH";
        case CUSPARSE_STATUS_MAPPING_ERROR: return "CUSPARSE_STATUS_MAPPING_ERROR";
        case CUSPARSE_STATUS_EXECUTION_FAILED: return "CUSPARSE_STATUS_EXECUTION_FAILED";
        case CUSPARSE_STATUS_INTERNAL_ERROR: return "CUSPARSE_STATUS_INTERNAL_ERROR";
    }
    return "unknown error";
}

inline const char* cusolverGetErrorString(cusolverStatus_t status)
{
    switch(status)
    {
        case CUSOLVER_STATUS_SUCCESS: return "CUSOLVER_STATUS_SUCCESS";
        case CUSOLVER_STATUS_NOT_INITIALIZED: return "CUSOLVER_STATUS_NOT_INITIALIZED";
        case CUSOLVER_STATUS_ALLOC_FAILED: return "CUSOLVER_STATUS_ALLOC_FAILED";
        case CUSOLVER_STATUS_INVALID_VALUE: return "CUSOLVER_STATUS_INVALID_VALUE";
        case CUSOLVER_STATUS_ARCH_MISMATCH: return "CUSOLVER_STATUS_ARCH_MISMATCH";
        case CUSOLVER_STATUS_MAPPING_ERROR: return "CUSOLVER_STATUS_MAPPING_ERROR";
        case CUSOLVER_STATUS_EXECUTION_FAILED: return "CUSOLVER_STATUS_EXECUTION_FAILED";
        case CUSOLVER_STATUS_INTERNAL_ERROR: return "CUSOLVER_STATUS_INTERNAL_ERROR";
    }
    return "unknown error";
}


inline static void HandleError( cudaError_t err,const char *file,int line ) {
    if (err != cudaSuccess) {
        printf( "CUDA error: %s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}

inline static void HandleError( cublasStatus_t err,const char *file,int line ) {
    if (err != CUBLAS_STATUS_SUCCESS) {
        printf( "CUBLAS error: %s in %s at line %d\n", cublasGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}

inline static void HandleError( cusparseStatus_t err,const char *file,int line ) {
    if (err != CUSPARSE_STATUS_SUCCESS) {
        printf( "CUSPARSE error: %s in %s at line %d\n", cusparseGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}
inline static void HandleError( cusolverStatus_t  err,const char *file,int line ) {
    if (err != CUSPARSE_STATUS_SUCCESS) {
        printf( "CUSOLVER error: %s in %s at line %d\n", cusolverGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}
inline void __getLastCudaError(const char *errorMessage, const char *file, const int line)
{
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
                file, line, errorMessage, (int)err, cudaGetErrorString(err));
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}
// This will output the proper error string when calling cudaGetLastError
#define CHECK_CUDA_ERROR(msg) __getLastCudaError (msg, __FILE__, __LINE__)

#define HANDLE_ERROR( err ) \
    cudaDeviceSynchronize(); \
    HandleError( err, __FILE__, __LINE__ )


#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}





class CudaHandlesKeeper
{
public:
    static CudaHandlesKeeper& instance()
    {
        static CudaHandlesKeeper    instance; // Guaranteed to be destroyed.
        return instance;
    }

    cublasHandle_t cublasHandle;
    cusparseHandle_t cusparseHandle;
    cusolverSpHandle_t cusolverSpHandle;

    AMGX_resources_handle amgxResource;
    AMGX_config_handle SPD_solver_config;
    AMGX_solver_handle SPD_solver_double;
	AMGX_solver_handle SPD_solver_single;


public:
    CudaHandlesKeeper(CudaHandlesKeeper const&)               = delete;
    void operator=(CudaHandlesKeeper const&)  = delete;

private:
    CudaHandlesKeeper() {
        HANDLE_ERROR(cublasCreate (& cublasHandle ));
        HANDLE_ERROR(cusparseCreate (& cusparseHandle ));
        HANDLE_ERROR(cusolverSpCreate(&cusolverSpHandle));

        AMGX_initialize();

        AMGX_initialize_plugins();
        AMGX_register_print_callback([](const char *msg, int length){
            //std::cout<<msg<<std::endl;
        });

        AMGX_install_signal_handler();

        AMGX_config_handle rsrc_config;
        AMGX_config_create(&rsrc_config, "");
        AMGX_resources_create_simple(&amgxResource, rsrc_config);

        AMGX_config_create_from_file(&SPD_solver_config, "./resources/AMGX-configs/AMG_FRANK.json");
        AMGX_solver_create(&SPD_solver_double, amgxResource, AMGX_mode_dDDI, SPD_solver_config);
		AMGX_solver_create(&SPD_solver_single, amgxResource, AMGX_mode_dFFI, SPD_solver_config);

    }
};



inline void printGLError(){
    GLenum err=glGetError();
    std::string error;
    switch(err) {
        case GL_INVALID_OPERATION:      error="INVALID_OPERATION";      break;
        case GL_INVALID_ENUM:           error="INVALID_ENUM";           break;
        case GL_INVALID_VALUE:          error="INVALID_VALUE";          break;
        case GL_OUT_OF_MEMORY:          error="OUT_OF_MEMORY";          break;

        case GL_INVALID_FRAMEBUFFER_OPERATION: error="GL_INVALID_FRAMEBUFFER_OPERATION"; break;
    }

    if(err!=GL_NO_ERROR){
        std::cerr <<"GL Error :  "<<error<<std::endl;
    }
}


inline void checkFramebufferComplete()
{
	GLenum err = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	const char* errString = NULL;
	switch (err) {
	case GL_FRAMEBUFFER_UNDEFINED: errString = "GL_FRAMEBUFFER_UNDEFINED"; break;
	case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT: errString = "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT"; break;
	case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT: errString = "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT"; break;
	case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER: errString = "GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER"; break;
	case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER: errString = "GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER"; break;
	case GL_FRAMEBUFFER_UNSUPPORTED: errString = "GL_FRAMEBUFFER_UNSUPPORTED"; break;
	case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE: errString = "GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE"; break;
	case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS: errString = "GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS"; break;
	}

	if (errString) {
		fprintf(stderr, "OpenGL Framebuffer Error #%d: %s\n", err, errString);
	}
}


inline void initOpenGL() {

	glfwInit();
	if (!glfwInit())
		exit(EXIT_FAILURE);

	// weird behavior under VS2019
	// if strspn && vsscanf is not used anywhere in the code
	// there will be a link error...
	// so they are used here



	char strtext[] = "129th";
	char cset[] = "1234567890";
	strspn(strtext, cset);

	auto GetMatches = [](const char* str, const char* format, ...) {
		va_list args;
		va_start(args, format);
		vsscanf(str, format, args);
		va_end(args);
	};

	int val;
	char buf[100];
	GetMatches("99 bottles of beer on the wall", " %d %s ", &val, buf);
}


inline void getScreenDimensions(int& width, int& height) {
	GLFWmonitor* monitor = glfwGetPrimaryMonitor();
	const GLFWvidmode* mode = glfwGetVideoMode(monitor);

	width = mode->width;
	height = mode->height;
}


inline GLFWwindow* createWindowOpenGL(int screenWidth,int screenHeight){
    GLFWwindow* window;
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    window = glfwCreateWindow(screenWidth, screenHeight, "Aquarius", nullptr, nullptr);
    glewExperimental = GL_TRUE;
    if (!window )
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(window);
    glewExperimental = GL_TRUE;
    if(glewInit() != GLEW_OK)
        throw std::runtime_error("glew Init failed");

    printGLError();
    glfwSwapInterval(1);

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    float ratio;
    ratio = width / (float) height;

    // Setup some OpenGL options
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glViewport(0, 0, width, height);
    return window;
}

inline float random0to1(){
    return (float)rand()/(float)RAND_MAX;
}

inline int divUp(int a,int b){
    int result =  (a % b != 0) ? (a/b + 1) : (a / b);
    return result;
}

#endif //AQUARIUS_GPUCOMMONS_H
