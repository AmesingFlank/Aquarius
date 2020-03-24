#pragma once

#include <vector>

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>



enum class CameraMovement {
   
	RotateUp,
	RotateDown,
	RotateRight,
	RotateLeft,

	MoveForward,
	MoveBackward,
	MoveRight,
	MoveLeft,
};

inline glm::vec3 eulerToVec(double yaw, double pitch) {
	glm::vec3 v;
	v.x = -sin(glm::radians(yaw)) * cos(glm::radians(pitch));
	v.y = sin(glm::radians(pitch));
	v.z = -cos(glm::radians(yaw)) * cos(glm::radians(pitch));
	return glm::normalize(v);
}


inline void vecToEuler(glm::vec3 v, double& yaw, double& pitch) {
	v = glm::normalize(v);
	pitch = glm::degrees(asin(v.y));

	double sinYaw = -v.x / cos(glm::radians(pitch));
	double cosYaw = -v.z / cos(glm::radians(pitch));

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

	double yaw;
    double pitch;

	double globalYaw;
	double globalPitch;
	float globalRadius = 18;

	float globalRotationSpeed = 1;
	float globalMovementSpeed = 0.3;
	float perspectiveRotationSpeed = 0.3;


	const float FOV = 45;

	glm::vec3 lookCenter;



    Camera(glm::vec3 lookCenter_) 
    {
		this->lookCenter = lookCenter_;

		yaw = 180;
        pitch = -30;

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
		globalRadius = glm::length(position);
	}

	void updateFrontWithEuler() {
		front = eulerToVec(yaw, pitch);
		updateRightUp();
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

    glm::mat4 getViewMatrix(){
        return glm::lookAt(getActualPosition(), getActualPosition() + front, up);
    }

	void clampPitch() {
		pitch = max(-89.0, min(89.0, pitch));
		globalPitch = max(-89.0, min(89.0, globalPitch));
	}


	void doMovement(CameraMovement move) {
		if (move == CameraMovement::RotateLeft) {
			globalYaw -= globalRotationSpeed;
			yaw -= globalRotationSpeed;
			updatePositionWithEuler();
			updateFrontWithEuler();
		}
		if (move == CameraMovement::RotateRight) {
			globalYaw += globalRotationSpeed;
			yaw += globalRotationSpeed;
			updatePositionWithEuler();
			updateFrontWithEuler();
		}
		if (move == CameraMovement::RotateUp) {
			globalPitch += globalRotationSpeed;
			pitch -= globalRotationSpeed;
			clampPitch();
			updatePositionWithEuler();
			updateFrontWithEuler();
		}
		if (move == CameraMovement::RotateDown) {
			globalPitch -= globalRotationSpeed;
			pitch += globalRotationSpeed;
			clampPitch();
			updatePositionWithEuler();
			updateFrontWithEuler();
		}
		if (move == CameraMovement::MoveLeft) {
			position -= right * globalMovementSpeed;
			updateEulerWithPosition();
		}
		if (move == CameraMovement::MoveRight) {
			position += right * globalMovementSpeed;
			updateEulerWithPosition();
		}
		if (move == CameraMovement::MoveBackward) {
			position -= front * globalMovementSpeed;
			updateEulerWithPosition();
		}
		if (move == CameraMovement::MoveForward) {
			position += front * globalMovementSpeed;
			updateEulerWithPosition();
		}
		
	}


	void perspectiveChange(float dx, float dy) {
		yaw -= dx * perspectiveRotationSpeed;
		pitch += dy * perspectiveRotationSpeed;
		clampPitch();
		updateFrontWithEuler();
	}

};
