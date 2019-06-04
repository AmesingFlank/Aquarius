//
// Created by AmesingFlank on 2019-04-17.
//

#ifndef AQUARIUS_CUDACOMMONS_H
#define AQUARIUS_CUDACOMMONS_H

#include <stdio.h>
#include <helper_math.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <cusolverSp.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <amgx_c.h>

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

static void HandleError( cublasStatus_t err,const char *file,int line ) {
    if (err != CUBLAS_STATUS_SUCCESS) {
        printf( "CUBLAS error: %s in %s at line %d\n", cublasGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}

static void HandleError( cusparseStatus_t err,const char *file,int line ) {
    if (err != CUSPARSE_STATUS_SUCCESS) {
        printf( "CUSPARSE error: %s in %s at line %d\n", cusparseGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}
static void HandleError( cusolverStatus_t  err,const char *file,int line ) {
    if (err != CUSPARSE_STATUS_SUCCESS) {
        printf( "CUSPARSE error: %s in %s at line %d\n", cusolverGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}


extern cublasHandle_t cublasHandle;
extern cusparseHandle_t cusparseHandle;
extern cusolverSpHandle_t cusolverSpHandle;
extern AMGX_resources_handle amgxResource;


inline void initCuda(){
    HANDLE_ERROR(cublasCreate (& cublasHandle ));
    HANDLE_ERROR(cusparseCreate (& cusparseHandle ));
    HANDLE_ERROR(cusolverSpCreate(&cusolverSpHandle));

    AMGX_initialize();

    AMGX_initialize_plugins();
    AMGX_register_print_callback([](const char *msg, int length){
        std::cout<<msg<<std::endl;
    });

    AMGX_install_signal_handler();

    AMGX_config_handle rsrc_config;
    AMGX_config_create(&rsrc_config, "");
    AMGX_resources_create_simple(&amgxResource, rsrc_config);

}



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


inline GLFWwindow* createWindowOpenGL(int screenWidth,int screenHeight){
    GLFWwindow* window;
    glfwInit();
    if (!glfwInit() )
        exit(EXIT_FAILURE);
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



#endif //AQUARIUS_CUDACOMMONS_H
