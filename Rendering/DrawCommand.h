#pragma once

#include <glm/glm.hpp>

enum class RenderMode:int {
	ScreenSpace = 0,Mesh = 1,Particles = 2,
	MAX = 3
};

struct DrawCommand {
	glm::mat4 view;
	glm::mat4 projection;
	glm::vec3 cameraPosition;
	float windowWidth;
	float windowHeight;
	float zoom;
	float near;
	float far;

	RenderMode renderMode;
};