#include <iostream>
#include <thrust/device_vector.h>
#include <chrono>
#include <thread>
#include <math.h>



#include "Common/GpuCommons.h"


#include "Fluid/Fluid_3D_FLIP.cuh"
#include "Fluid/Fluid_3D_PCISPH.cuh"
#include "Fluid/Fluid_3D_PBF.cuh"
#include "Fluid/Fluid_3D.cuh"




#include "Rendering/Renderer3D/camera.h"
#include "Common/InputHandler.h"
#include "Rendering/Renderer3D/PointSprites.h"

#include "Rendering/DrawCommand.h"


#include "Rendering/WindowInfo.h"
#include "Fluid/FluidConfig.cuh"

#include "UI/ui.h"


int main( void ) {

	initOpenGL();

    int screenWidth;
    int screenHeight;

	getScreenDimensions(screenWidth, screenHeight);

	WindowInfo& windowInfo = WindowInfo::instance();

	windowInfo.windowWidth = screenWidth * 0.9;
	windowInfo.windowHeight = windowInfo.windowWidth / 2;

    GLFWwindow* window = createWindowOpenGL(windowInfo.windowWidth, windowInfo.windowHeight);

	nk_context* uiContext = createUI(window);

    glfwSetKeyCallback(window, InputHandler::key_callback);
    glfwSetCursorPosCallback(window, InputHandler::mouse_callback);

    std::shared_ptr<Camera> camera = std::make_shared<Camera>(glm::vec3(5,10,20));
    InputHandler::camera = camera;

	
	double framesSinceLast = 0;
	double lastSecond = glfwGetTime();
	double lastFrameTime = glfwGetTime();

	glEnable(GL_BLEND);


	FluidConfig config;
	std::shared_ptr<Fluid_3D> fluid;

	

	RenderMode renderMode = RenderMode::Mesh;

	bool paused = true;

	bool hasCreatedFluid = false;

	

    while(!glfwWindowShouldClose(window)){


        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glClear(GL_COLOR_BUFFER_BIT);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
        glClearColor(0,0,0,1);
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

		

        double currentTime = glfwGetTime();

		if (hasCreatedFluid) {
			glm::vec2 fluidCenter = fluid->getCenter();

			glm::vec3 lightPos(fluidCenter.x, 30, fluidCenter.y);

			DrawCommand drawCommand = {
			view,projection,camera->Position,windowInfo.windowWidth,windowInfo.windowHeight,camera->Zoom,near,far,
			renderMode,paused,lightPos
			};

			if (!paused) {
				fluid->simulationStep();
			}

			fluid->draw(drawCommand);
		}

		

		drawUI(uiContext,config, [&]() 
			{
				if (hasCreatedFluid) {
					fluid.reset();
				}
				if (config.method == "FLIP") {
					fluid = std::static_pointer_cast<Fluid_3D, Fluid_3D_FLIP::Fluid>(std::make_shared<Fluid_3D_FLIP::Fluid>());
				}
				else if (config.method == "PCISPH") {
					fluid = std::static_pointer_cast<Fluid_3D, Fluid_3D_PCISPH::Fluid>(std::make_shared<Fluid_3D_PCISPH::Fluid>());
				}
				else if (config.method == "PBF") {
					fluid = std::static_pointer_cast<Fluid_3D, Fluid_3D_PBF::Fluid>(std::make_shared<Fluid_3D_PBF::Fluid>());
				}
				else {
					std::cout << "unsupported method in config file" << std::endl;
					exit(1);
				}

				fluid->init(config);
				hasCreatedFluid = true;
			}
		);

        ++framesSinceLast;

        if(currentTime-lastSecond>=1){
            double FPS = (double)framesSinceLast/(currentTime-lastSecond);
            std::cout<<"FPS: "<<FPS<<std::endl;
			std::string fpsText = "Aquarius  " + std::to_string(FPS) + " FPS";
			glfwSetWindowTitle(window, fpsText.c_str());
            lastSecond = currentTime;
            framesSinceLast = 0;
        }

        lastFrameTime = currentTime;

        printGLError();
        glfwSwapBuffers(window);

        //break;
    }

	fluid.reset();
	printGLError();

    std::cout<<"finished everything"<<std::endl;
	cudaDeviceReset();

    return 0;
}
