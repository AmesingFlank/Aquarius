#include <iostream>
#include <thrust/device_vector.h>
#include <chrono>
#include <thread>
#include <math.h>



#include "GpuCommons.h"

/*
#include "Fluid_2D_SemiLagrange.cuh"
#include "Fluid_2D_PCISPH.cuh"
#include "Fluid_2D_PCISPH_CPU.h"
#include "Fluid_2D_PoistionBased_CPU.h"
#include "Fluid_2D_Full.cuh"
#include "Fluid_3D_SPH.cuh"
#include "Fluid_3D_PCISPH.cuh"
*/

#include "Fluid/Fluid_2D_FLIP.cuh"


#include "Rendering/Renderer3D/Renderer3D.h"
#include "Rendering/Renderer3D/camera.h"
#include "InputHandler.h"
#include "Rendering/Renderer3D/PointSprites.h"

#include "Rendering/DrawCommand.h"






int main( void ) {

	initOpenGL();

    int screenWidth;
    int screenHeight;

	getScreenDimensions(screenWidth, screenHeight);

	float windowWidth = screenWidth / 2;
	float windowHeight = windowWidth / 2;

    GLFWwindow* window = createWindowOpenGL(windowWidth,windowHeight);

    glfwSetKeyCallback(window, InputHandler::key_callback);
    glfwSetCursorPosCallback(window, InputHandler::mouse_callback);

    std::shared_ptr<Camera> camera = std::make_shared<Camera>(glm::vec3(5,10,20));
    InputHandler::camera = camera;

    Fluid_2D_FLIP::Fluid fluid;
	//Fluid_3D_PCISPH::Fluid fluid;


	double framesSinceLast = 0;
	double lastSecond = glfwGetTime();
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
        glm::mat4 projection = glm::perspective(camera->Zoom, (float)windowWidth/(float)windowHeight, 0.1f, 10000.0f);

		DrawCommand drawCommand = {
			view,projection,camera->Position,windowWidth,windowHeight
		};


        double currentTime = glfwGetTime();

        fluid.simulationStep();
        fluid.draw(drawCommand);

        //skybox.draw(view,projection);
		//fluid.simulationStep();
		//fluid.draw(view,projection,camera->Position,windowWidth,windowHeight);

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
