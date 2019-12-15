#pragma once

#include "../../GpuCommons.h"
#include "../Shader.h"
#include "../DrawCommand.h"

struct FluidMeshRenderer {
	int count;
	const int floatsPerVertex = 3;
	float* coordsHost;
	float* coordsDevice;
	GLuint VAO, VBO;
	GLint vPos_location;
	cudaGraphicsResource* cudaResourceVBO;
	Shader* shader;

	FluidMeshRenderer(int count_);
	void draw(const DrawCommand& drawCommand);
};