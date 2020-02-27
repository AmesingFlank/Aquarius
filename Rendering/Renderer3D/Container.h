#pragma once

#include "../DrawCommand.h"
#include "../../Common/GpuCommons.h"
#include "../Shader.h"

struct Container {
	float edgesData[12 * 6] = {
		0,0,0, 0,0,1,
		0,1,0, 0,1,1,
		1,0,0, 1,0,1,
		1,1,0, 1,1,1,

		0,0,0, 0,1,0,
		0,0,1, 0,1,1,
		1,0,0, 1,1,0,
		1,0,1, 1,1,1,

		0,0,0, 1,0,0,
		0,0,1, 1,0,1,
		0,1,0, 1,1,0,
		0,1,1, 1,1,1,
	};

	GLuint edgesVBO, edgesVAO;

	std::shared_ptr<Shader> edgesShader;

	float minVal = -1000;
	float maxVal = 1000;

	float bottomData[2*3*3] = {
		minVal,0,minVal, minVal,0,maxVal, maxVal,0,maxVal,

		minVal,0,minVal, maxVal,0,maxVal, maxVal,0,minVal
	};

	GLuint bottomVBO, bottomVAO;

	std::shared_ptr<Shader> bottomShader;

	glm::mat4 model = glm::mat4(1.0);

	void initEdges();

	void drawEdges(const DrawCommand& drawCommand);

	void initBottom();

	void drawBottom(const DrawCommand& drawCommand);

	void draw(const DrawCommand& drawCommand);

	Container(float size);
};