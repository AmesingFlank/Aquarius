
#ifndef AQUARIUS_INPUTHANDLER_H
#define AQUARIUS_INPUTHANDLER_H

#include <memory>
#include "../Rendering/Renderer3D/camera.h"
#include "GpuCommons.h"

namespace InputHandler{

    //Singleton
	class Handler {
	public:
		static Handler& instance()
		{
			static Handler   instance; 
			return instance;
		}

		bool keys[1024];
		float lastX = 0;
		float lastY = 0;
		bool firstMouse = true;
		std::shared_ptr<Camera> camera;

		bool leftMouseDown = false;

		std::function<void()> onShift = []() {};
		std::function<void()> onSpace = []() {};
		std::function<void()> onEscape = []() {};
		std::function<void(int key)> onKeyGeneral = [](int key) {};

		void doMovement()
		{
			if (!camera) {
				return;
			}
			if (keys[GLFW_KEY_LEFT]) {
				camera->doMovement(CameraMovement::RotateLeft);
			}
			if (keys[GLFW_KEY_RIGHT]) {
				camera->doMovement(CameraMovement::RotateRight);
			}
			if (keys[GLFW_KEY_UP]) {
				camera->doMovement(CameraMovement::RotateUp);
			}
			if (keys[GLFW_KEY_DOWN]) {
				camera->doMovement(CameraMovement::RotateDown);
			}
			if (keys[GLFW_KEY_A]) {
				camera->doMovement(CameraMovement::MoveLeft);
			}
			if (keys[GLFW_KEY_D]) {
				camera->doMovement(CameraMovement::MoveRight);
			}
			if (keys[GLFW_KEY_W]) {
				camera->doMovement(CameraMovement::MoveForward);
			}
			if (keys[GLFW_KEY_S]) {
				camera->doMovement(CameraMovement::MoveBackward);
			}
		}

		void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mode)
		{
			if (action == GLFW_PRESS) {
				if (key == GLFW_KEY_ESCAPE) {
					onEscape();
				}
				if (key == GLFW_KEY_LEFT_SHIFT || key==GLFW_KEY_RIGHT_SHIFT) {
					onShift();
				}
				if (key == GLFW_KEY_SPACE) {
					onSpace();
				}
				onKeyGeneral(key);
			}

			if (action == GLFW_PRESS) {
				keys[key] = true;

			}
			else if (action == GLFW_RELEASE) {
				keys[key] = false;
			}
		}

		void mousePosCallback(GLFWwindow* window, double xpos, double ypos)
		{
			if (firstMouse)
			{
				lastX = xpos;
				lastY = ypos;
				firstMouse = false;
			}

			GLfloat dx = xpos - lastX;
			GLfloat dy = lastY - ypos;

			lastX = xpos;
			lastY = ypos;

			if (camera && leftMouseDown) {
				camera->perspectiveChange(dx, dy);
			}
		}

		void mouseButtonCallback(GLFWwindow* window, int button, int action, int modifier)
		{
			if (button == GLFW_MOUSE_BUTTON_LEFT) {
				if (action == GLFW_PRESS) {
					leftMouseDown = true;
					glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
				}
				if (action == GLFW_RELEASE) {
					leftMouseDown = false;
					glfwSetInputMode(window, GLFW_CURSOR,GLFW_CURSOR_NORMAL);
				}
			}
		}

	public:
		Handler(Handler const&) = delete;
		void operator=(Handler const&) = delete;

	private:
		Handler() {


		}
	};
    


    inline void doMovement()
    {
		Handler::instance().doMovement();
    }

    inline void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mode)
    {
		Handler::instance().keyCallback(window, key, scancode, action, mode);
    }

    inline void mousePosCallback(GLFWwindow* window, double xpos, double ypos)
    {
		Handler::instance().mousePosCallback(window, xpos,ypos);
    }

	inline void mouseButtonCallback(GLFWwindow* window, int button, int action, int modifier)
	{
		Handler::instance().mouseButtonCallback(window, button,action,modifier);
	}

	


};

#endif //AQUARIUS_INPUTHANDLER_H
