
#ifndef AQUARIUS_INPUTHANDLER_H
#define AQUARIUS_INPUTHANDLER_H

#include <memory>
#include "Rendering/Renderer3D/camera.h"
#include "GpuCommons.h"

namespace InputHandler{

    
	class Handler {
	public:
		static Handler& instance()
		{
			static Handler   instance; // Guaranteed to be destroyed.
			return instance;
		}

		bool keys[1024];
		GLfloat lastX = 400, lastY = 300;
		bool firstMouse = true;
		std::shared_ptr<Camera> camera;

		void Do_Movement()
		{
			// Camera controls
			if (keys[GLFW_KEY_W])
				camera->ProcessKeyboard(FORWARD);
			if (keys[GLFW_KEY_S])
				camera->ProcessKeyboard(BACKWARD);
			if (keys[GLFW_KEY_A])
				camera->ProcessKeyboard(LEFT);
			if (keys[GLFW_KEY_D])
				camera->ProcessKeyboard(RIGHT);
		}

		void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode)
		{
			if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
				glfwSetWindowShouldClose(window, GL_TRUE);

			if (action == GLFW_PRESS)
				keys[key] = true;
			else if (action == GLFW_RELEASE) {
				keys[key] = false;
				camera->lastProcessKeyboardTime = -1;
			}
		}

		void mouse_callback(GLFWwindow* window, double xpos, double ypos)
		{
			if (firstMouse)
			{
				lastX = xpos;
				lastY = ypos;
				firstMouse = false;
			}

			GLfloat xoffset = xpos - lastX;
			GLfloat yoffset = lastY - ypos;

			lastX = xpos;
			lastY = ypos;

			camera->ProcessMouseMovement(xoffset, yoffset);
		}

	public:
		Handler(Handler const&) = delete;
		void operator=(Handler const&) = delete;

	private:
		Handler() {


		}
	};
    


    inline void Do_Movement()
    {
		Handler::instance().Do_Movement();
    }

    inline void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode)
    {
		Handler::instance().key_callback(window, key, scancode, action, mode);
    }

    inline void mouse_callback(GLFWwindow* window, double xpos, double ypos)
    {
		Handler::instance().mouse_callback(window, xpos,ypos);
    }

	


};

#endif //AQUARIUS_INPUTHANDLER_H
