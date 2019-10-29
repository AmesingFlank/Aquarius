//
// Created by AmesingFlank on 2019-08-13.
//

#ifndef AQUARIUS_INPUTHANDLER_H
#define AQUARIUS_INPUTHANDLER_H

#include <memory>
#include "Rendering/Renderer3D/camera.h"
#include "GpuCommons.h"

namespace InputHandler{

    bool keys[1024];
    GLfloat lastX = 400, lastY = 300;
    bool firstMouse = true;

    std::shared_ptr<Camera> camera;


// Moves/alters the camera positions based on user input
    inline void Do_Movement()
    {
        // Camera controls
        if(keys[GLFW_KEY_W])
            camera->ProcessKeyboard(FORWARD);
        if(keys[GLFW_KEY_S])
            camera->ProcessKeyboard(BACKWARD);
        if(keys[GLFW_KEY_A])
            camera->ProcessKeyboard(LEFT);
        if(keys[GLFW_KEY_D])
            camera->ProcessKeyboard(RIGHT);
    }

// Is called whenever a key is pressed/released via GLFW
    inline void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode)
    {
        if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
            glfwSetWindowShouldClose(window, GL_TRUE);

        if(action == GLFW_PRESS)
            keys[key] = true;
		else if (action == GLFW_RELEASE) {
			keys[key] = false;
			camera->lastProcessKeyboardTime = -1;
		}
    }

    inline void mouse_callback(GLFWwindow* window, double xpos, double ypos)
    {
        if(firstMouse)
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


};

#endif //AQUARIUS_INPUTHANDLER_H
