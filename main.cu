#include <iostream>
#include <thrust/device_vector.h>
#include <chrono>
#include <thread>
#include <math.h>


#include "SPD_Solver.h"
#include "Rendering/Quad.h"
#include "Fluid_2D_SemiLagrange.cuh"
#include "Fluid_2D_PCISPH.cuh"
#include "GpuCommons.h"
#include "Fluid_2D_PCISPH_CPU.h"
#include "Fluid_2D_PoistionBased_CPU.h"
#include "Fluid_2D_FLIP.cuh"

#include "MAC_Grid_3D.cuh"

#include "Rendering/Renderer3D/Renderer3D.h"
#include "Rendering/Renderer3D/camera.h"
#include "InputHandler.h"






int main( void ) {

	initOpenGL();

    int screenWidth;
    int screenHeight;

	getScreenDimensions(screenWidth, screenHeight);

    GLFWwindow* window = createWindowOpenGL(screenWidth/2,screenWidth/2/2);

    glfwSetKeyCallback(window, InputHandler::key_callback);
    glfwSetCursorPosCallback(window, InputHandler::mouse_callback);

    std::shared_ptr<Camera> camera = std::make_shared<Camera>(glm::vec3(0.0f, 0.0f, 3.0f));
    InputHandler::camera = camera;

    Fluid_2D_FLIP::Fluid fluid;
    Quad quad;

    Skybox skybox("./resources/Park2/",".jpg");

    double framesSinceLast = 0;
    double lastSecond=glfwGetTime();
    double lastFrameTime = glfwGetTime();

    while(!glfwWindowShouldClose(window)){

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glClear(GL_COLOR_BUFFER_BIT);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
        glClearColor(0.25f,0.38f,0.5f,0.f);
        glfwPollEvents();
        InputHandler::Do_Movement();

        glm::mat4 view = camera->GetViewMatrix();
        glm::mat4 projection = glm::perspective(camera->Zoom, (float)screenWidth/(float)screenHeight, 0.1f, 10000.0f);


        double currentTime = glfwGetTime();

        fluid.simulationStep(currentTime-lastFrameTime);
        fluid.updateTexture();
        quad.draw(fluid.texture);

        //skybox.draw(view,projection);

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

        //break;
    }

    std::cout<<"finished everything"<<std::endl;

    return 0;
}
