#pragma once

#include <vector>

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>



enum class CameraMovement {
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT,
	RotateUp,
	RotateDown,
	RotateRight,
	RotateLeft
};


const GLfloat SPEED      =  3.0f;
const GLfloat SENSITIVTY =  0.25f;


inline glm::vec3 eulerToVec(float yaw, float pitch) {
	glm::vec3 v;
	v.x = -sin(glm::radians(yaw)) * cos(glm::radians(pitch));
	v.y = sin(glm::radians(pitch));
	v.z = -cos(glm::radians(yaw)) * cos(glm::radians(pitch));
	return glm::normalize(v);
}


inline void vecToEuler(glm::vec3 v, float& yaw, float& pitch) {
	v = glm::normalize(v);
	pitch = glm::degrees(asin(v.y));

	float sinYaw = -v.x / cos(glm::radians(pitch));
	float cosYaw = -v.z / cos(glm::radians(pitch));

	yaw = acos(cosYaw);
	if (sinYaw < 0) {
		yaw = -yaw;
	}

	yaw = glm::degrees(yaw);

}

inline void printVec3(glm::vec3 v) {
	std::cout << v.x << " " << v.y << " " << v.z << std::endl;
}

class Camera
{
public:

	glm::vec3 position;
    glm::vec3 front;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec3 worldUp = glm::vec3(0.0f, 1.0f, 0.0f);

	float yaw;
    float pitch;

	float globalYaw;
	float globalPitch;
	float globalRadius = 20;

	float globalRotationSpeed = 1;

	GLfloat MovementSpeed;
    GLfloat MouseSensitivity;

	const float FOV = 45;

	glm::vec3 lookCenter;



    Camera(glm::vec3 lookCenter_) :  MovementSpeed(SPEED), MouseSensitivity(SENSITIVTY)
    {
		this->lookCenter = lookCenter_;

		yaw = 0;
        pitch = 0;

		globalYaw = 0;
		globalPitch = 30;

		updatePositionWithEuler();
		lookAtCenter();

		printVec3(getActualPosition());
		printVec3(position);
		printVec3(front);
		printVec3(up);
		std::cout << globalYaw << " " << globalPitch << std::endl << std::endl;
    } 

	void updatePositionWithEuler() {
		position = eulerToVec(globalYaw , globalPitch) * globalRadius;
	}

	void updateEulerWithPosition() {
		vecToEuler(position, globalYaw, globalPitch);
	}

	void lookAtCenter() {
		front = glm::normalize(-position);
		updateRightUp();
	}

	void updateRightUp() {
		this->right = glm::normalize(glm::cross(this->front, this->worldUp));
		this->up = glm::normalize(glm::cross(this->right, this->front));
	}

	// The camera treats lookPosition and the center of world coordinates
	glm::vec3 getActualPosition() {
		return lookCenter + position;
	}

    glm::mat4 getViewMatrix()
    {
		
        return glm::lookAt(getActualPosition(), getActualPosition() + front, up);
    }



	void doMovement(CameraMovement move) {
		if (move == CameraMovement::RotateLeft) {
			globalYaw -= globalRotationSpeed;
		}
		if (move == CameraMovement::RotateRight) {
			globalYaw += globalRotationSpeed;
		}
		if (move == CameraMovement::RotateUp) {
			globalPitch += globalRotationSpeed;
		}
		if (move == CameraMovement::RotateDown) {
			globalPitch -= globalRotationSpeed;
		}
		updatePositionWithEuler();
		lookAtCenter();
	}

	float lastProcessKeyboardTime = -1;

	void ProcessKeyboard(CameraMovement direction)
    {
        float now = glfwGetTime();
        if(lastProcessKeyboardTime<0){
            lastProcessKeyboardTime = now;
        }
        float deltaTime = now-lastProcessKeyboardTime;
        GLfloat velocity = this->MovementSpeed * deltaTime;
        if (direction == CameraMovement::FORWARD)
            this->position += this->front * velocity;
        if (direction == CameraMovement::BACKWARD)
            this->position -= this->front * velocity;
        if (direction == CameraMovement::LEFT)
            this->position -= this->right * velocity;
        if (direction == CameraMovement::RIGHT)
            this->position += this->right * velocity;
        lastProcessKeyboardTime = now;
		//std::cout << "camera at " << position.x << "  " << position.y <<"  "<< position.z << std::endl;
    }

    // Processes input received from a mouse input system. Expects the offset value in both the x and y direction.
    void ProcessMouseMovement(GLfloat xoffset, GLfloat yoffset, GLboolean constrainPitch = true)
    {
        xoffset *= this->MouseSensitivity;
        yoffset *= this->MouseSensitivity;

        this->yaw   -= xoffset;
        this->pitch += yoffset;

        // Make sure that when pitch is out of bounds, screen doesn't get flipped
        if (constrainPitch)
        {
            if (this->pitch > 89.0f)
                this->pitch = 89.0f;
            if (this->pitch < -89.0f)
                this->pitch = -89.0f;
        }

        // Update front, right and up Vectors using the updated Eular angles
        this->updateCameraVectors();
    }



private:
    // Calculates the front vector from the Camera's (updated) Eular Angles
    void updateCameraVectors()
    {

        this->front = eulerToVec(yaw,pitch);

		if (abs(glm::length(front) - 1) > 1e-3) {
			std::cout << "wat??" << std::endl;
		}

        // Also re-calculate the right and up vector
        this->right = glm::normalize(glm::cross(this->front, this->worldUp)); 
        this->up    = glm::normalize(glm::cross(this->right, this->front));
    }
};
