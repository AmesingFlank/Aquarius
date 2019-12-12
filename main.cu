#include <iostream>
#include <thrust/device_vector.h>
#include <chrono>
#include <thread>
#include <math.h>



#include "GpuCommons.h"


#include "Fluid/Fluid_3D_FLIP.cuh"



#include "Rendering/Renderer3D/Renderer3D.h"
#include "Rendering/Renderer3D/camera.h"
#include "InputHandler.h"
#include "Rendering/Renderer3D/PointSprites.h"

#include "Rendering/DrawCommand.h"


#include "Rendering/WindowInfo.h"



int main( void ) {

	initOpenGL();

    int screenWidth;
    int screenHeight;

	getScreenDimensions(screenWidth, screenHeight);

	WindowInfo& windowInfo = WindowInfo::instance();

	windowInfo.windowWidth = screenWidth * 3 / 4;
	windowInfo.windowHeight = windowInfo.windowWidth / 2;

    GLFWwindow* window = createWindowOpenGL(windowInfo.windowWidth, windowInfo.windowHeight);

    glfwSetKeyCallback(window, InputHandler::key_callback);
    glfwSetCursorPosCallback(window, InputHandler::mouse_callback);

    std::shared_ptr<Camera> camera = std::make_shared<Camera>(glm::vec3(5,10,20));
    InputHandler::camera = camera;

    Fluid_3D_FLIP::Fluid fluid;
	
	double framesSinceLast = 0;
	double lastSecond = glfwGetTime();
	double lastFrameTime = glfwGetTime();

	glEnable(GL_BLEND);

    while(!glfwWindowShouldClose(window)){

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glClear(GL_COLOR_BUFFER_BIT);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
        glClearColor(0.25f,0.38f,0.5f,0.f);
        glfwPollEvents();
        InputHandler::Do_Movement();

		float near = 0.1;
		float far = 1000;

        glm::mat4 view = camera->GetViewMatrix();
		float widthHeightRatio = (float)windowInfo.windowWidth / (float)windowInfo.windowHeight;
        glm::mat4 projection = glm::perspective(camera->Zoom, widthHeightRatio, near,far);

		DrawCommand drawCommand = {
			view,projection,camera->Position,windowInfo.windowWidth,windowInfo.windowHeight,camera->Zoom,near,far
		};


        double currentTime = glfwGetTime();

		fluid.simulationStep();
		fluid.draw(drawCommand);

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
	cudaDeviceReset();

    return 0;
}
