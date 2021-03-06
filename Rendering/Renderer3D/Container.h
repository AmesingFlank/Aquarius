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

	float minVal = -1;
	float maxVal = 2;

	float bottomData[108] = {
			0.0f, 0.0f, 0.0f,
			0.0f, 0.0f,  1.0f,
			1.0f, 0.0f, 0.0f,
			1.0f, 0.0f, 0.0f,
			0.0f, 0.0f,  1.0f,
			1.0f, 0.0f,  1.0f,

			0.0f,  1.0f, 0.0f,
			0.0f, 0.0f, 0.0f,
			1.0f, 0.0f, 0.0f,
			1.0f, 0.0f, 0.0f,
			1.0f,  1.0f, 0.0f,
			0.0f,  1.0f, 0.0f,

			0.0f, 0.0f,  1.0f,
			0.0f, 0.0f, 0.0f,
			0.0f,  1.0f, 0.0f,
			0.0f,  1.0f, 0.0f,
			0.0f,  1.0f,  1.0f,
			0.0f, 0.0f,  1.0f,

			1.0f, 0.0f, 0.0f,
			1.0f, 0.0f,  1.0f,
			1.0f,  1.0f,  1.0f,
			1.0f,  1.0f,  1.0f,
			1.0f,  1.0f, 0.0f,
			1.0f, 0.0f, 0.0f,

			0.0f, 0.0f,  1.0f,
			0.0f,  1.0f,  1.0f,
			1.0f,  1.0f,  1.0f,
			1.0f,  1.0f,  1.0f,
			1.0f, 0.0f,  1.0f,
			0.0f, 0.0f,  1.0f,

			0.0f,  1.0f, 0.0f,
			1.0f,  1.0f, 0.0f,
			1.0f,  1.0f,  1.0f,
			1.0f,  1.0f,  1.0f,
			0.0f,  1.0f,  1.0f,
			0.0f,  1.0f, 0.0f,

			
	};

	GLuint bottomVBO, bottomVAO;

	GLuint texOxLogo;

	std::shared_ptr<Shader> faceShader;

	glm::mat4 model = glm::mat4(1.0);

	void loadOxLogo();

	void initEdges();

	void drawEdges(const DrawCommand& drawCommand);

	void initBottom();

	void drawFace(const DrawCommand& drawCommand);

	void draw(const DrawCommand& drawCommand);

	Container(float size);

	float size;

	float cornellBoxSize;

	float bigChessBoardSize;

	~Container();
};