#pragma once

#include <glm/glm.hpp>
struct DrawCommand {
	glm::mat4 view;
	glm::mat4 projection;
	glm::vec3 cameraPosition;
	float windowWidth;
	float windowHeight;
};