#include <iostream>
#include <thrust/device_vector.h>
#include <chrono>
#include <thread>
#include <math.h>



#include "GpuCommons.h"


#include "Fluid/Fluid_3D_FLIP.cuh"
#include "Fluid/Fluid_3D_PCISPH.cuh"



#include "Rendering/Renderer3D/camera.h"
#include "InputHandler.h"
#include "Rendering/Renderer3D/PointSprites.h"

#include "Rendering/DrawCommand.h"


#include "Rendering/WindowInfo.h"
#include "Fluid/FluidConfig.cuh"


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

	
	double framesSinceLast = 0;
	double lastSecond = glfwGetTime();
	double lastFrameTime = glfwGetTime();

	glEnable(GL_BLEND);


	std::shared_ptr<FluidConfig> config = getConfig();
	std::shared_ptr<Fluid> fluid;
	if (config->method == "Fluid_3D_FLIP") {
		fluid = std::static_pointer_cast<Fluid,Fluid_3D_FLIP::Fluid>(std::make_shared<Fluid_3D_FLIP::Fluid>());
	}
	else if (config->method == "Fluid_3D_PCISPH") {
		fluid = std::static_pointer_cast<Fluid, Fluid_3D_PCISPH::Fluid>(std::make_shared<Fluid_3D_PCISPH::Fluid>());
	}
	else {
		std::cout << "unsupported method in config file" << std::endl;
		exit(1);
	}

	fluid->init(config);

	RenderMode renderMode = RenderMode::Mesh;

	bool paused = false;

    while(!glfwWindowShouldClose(window)){


        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glClear(GL_COLOR_BUFFER_BIT);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
        glClearColor(0.25f,0.38f,0.5f,0.f);
        glfwPollEvents();
        InputHandler::Do_Movement();
		if (InputHandler::keys[GLFW_KEY_SPACE]) {
			InputHandler::keys[GLFW_KEY_SPACE] = false;
			paused = !paused;
		}
		if (InputHandler::keys[GLFW_KEY_RIGHT_SHIFT]) {
			InputHandler::keys[GLFW_KEY_RIGHT_SHIFT] = false;
			renderMode = (RenderMode)(((int)renderMode + 1) % (int)RenderMode::MAX);
		}

		float near = 0.1;
		float far = 1000;

        glm::mat4 view = camera->GetViewMatrix();
		float widthHeightRatio = (float)windowInfo.windowWidth / (float)windowInfo.windowHeight;
        glm::mat4 projection = glm::perspective(camera->Zoom, widthHeightRatio, near,far);

		DrawCommand drawCommand = {
			view,projection,camera->Position,windowInfo.windowWidth,windowInfo.windowHeight,camera->Zoom,near,far,
			renderMode
		};


        double currentTime = glfwGetTime();

		if (!paused) {
			fluid->simulationStep();
		}
		else {
			// There's still a bug that, when paused, the meshed rendering doesn't work, unless sleep for a while..
			std::this_thread::sleep_for(std::chrono::milliseconds(16));
		}
		
		fluid->draw(drawCommand);

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
