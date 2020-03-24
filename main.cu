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

// TODO:
// Improve Reflection/Refraction Calculation
// Leave Comments
// Convert To CMAKE
// ScreenShots
// Videos
// Update README


int main( void ) {

	initOpenGL();

    int screenWidth;
    int screenHeight;

	getScreenDimensions(screenWidth, screenHeight);

	WindowInfo& windowInfo = WindowInfo::instance();
	InputHandler::Handler& inputHandler = InputHandler::Handler::instance();

	windowInfo.windowWidth = screenWidth * 0.9;
	windowInfo.windowHeight = windowInfo.windowWidth / 2;

    GLFWwindow* window = createWindowOpenGL(windowInfo.windowWidth, windowInfo.windowHeight);

	nk_context* uiContext = createUI(window);

    glfwSetKeyCallback(window, InputHandler::keyCallback);
    glfwSetCursorPosCallback(window, InputHandler::mousePosCallback);
	glfwSetMouseButtonCallback(window, InputHandler::mouseButtonCallback);

	std::shared_ptr<Camera> camera;
	
	double framesSinceLast = 0;
	double lastSecond = glfwGetTime();
	double lastFrameTime = glfwGetTime();

	glEnable(GL_BLEND);


	FluidConfig config;
	std::shared_ptr<Fluid_3D> fluid;


	EnvironmentMode environmentMode = EnvironmentMode::Skybox;
	Skybox skybox;
	std::shared_ptr<Container> container;
	

	RenderMode renderMode = RenderMode::MultiphaseMesh;

	bool paused = true;

	bool hasCreatedFluid = false;

	inputHandler.onEscape = [&]() {
		glfwSetWindowShouldClose(window, GL_TRUE);
	};
	inputHandler.onShift = [&]() {
		renderMode = (RenderMode)(((int)renderMode + 1) % (int)RenderMode::MAX);
	};
	inputHandler.onSpace = [&]() {
		paused = !paused;
	};
	inputHandler.onKeyGeneral = [&](int key) {
		if (key == GLFW_KEY_E) {
			environmentMode = (EnvironmentMode)(((int)environmentMode + 1) % (int)EnvironmentMode::MAX);
		}
	};


	
    while(!glfwWindowShouldClose(window)){


        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glClear(GL_COLOR_BUFFER_BIT);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
        glClearColor(0,0,0,1);
        glfwPollEvents();

        InputHandler::doMovement();


        double currentTime = glfwGetTime();

		if (hasCreatedFluid) {

			float near = 0.1;
			float far = 1000;

			glm::mat4 view = camera->getViewMatrix();
			float widthHeightRatio = (float)windowInfo.windowWidth / (float)windowInfo.windowHeight;
			glm::mat4 projection = glm::perspective(camera->FOV, widthHeightRatio, near, far);

			glm::vec3 fluidCenter = fluid->getCenter();

			glm::vec3 lightPosition(fluidCenter.x, 15, fluidCenter.y);

			DrawCommand drawCommand = {
			view,projection,camera->getActualPosition(),windowInfo.windowWidth,windowInfo.windowHeight,camera->FOV,near,far,
			renderMode,paused,lightPosition,environmentMode,skybox.texSkyBox,container->size,container->cornellBoxSize
			};

			if (!paused) {
				fluid->simulationStep();
			}

			if (environmentMode == EnvironmentMode::Skybox) {
				skybox.draw(drawCommand);
			}
			container->drawFace(drawCommand);
			fluid->draw(drawCommand);
		}

		

		drawUI(uiContext,config, [&]() 
			{
				if (config.initialVolumes.size() == 0) {
					std::cout << "ERROR: No Initial Volumes" << std::endl;
					return;
				}

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
				
				

				fluid->init(config);
				hasCreatedFluid = true;

				camera = std::make_shared<Camera>(fluid->getCenter());
				container = std::make_shared<Container>(fluid->getContainerSize());
				inputHandler.camera = camera;
				paused = true;
			}
		);

        ++framesSinceLast;

        if(currentTime-lastSecond>=1){
            double FPS = (double)framesSinceLast/(currentTime-lastSecond);
            std::cout<<"FPS: "<<FPS<<std::endl;
			std::string windowText = "Aquarius  " + std::to_string(FPS) + " FPS";
			if (fluid) {
				windowText += "             t:" + std::to_string(fluid->physicalTime);
			}
			glfwSetWindowTitle(window, windowText.c_str());
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
