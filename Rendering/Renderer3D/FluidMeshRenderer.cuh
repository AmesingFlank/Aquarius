#pragma once

#include "../../Common/GpuCommons.h"
#include "../Shader.h"
#include "../DrawCommand.h"
#include "ScreenSpaceNormal.cuh"

struct FluidMeshRenderer {
	int count;
	const int floatsPerVertex = 6;
	float* coordsHost;
	float* coordsDevice;
	GLuint VAO, VBO;
	GLint positionLocation;
	GLint normalLocation;
	cudaGraphicsResource* cudaResourceVBO;
	Shader* shader;
	Shader* depthShader;

	ScreenSpaceNormal screenSpaceNormal;

	FluidMeshRenderer(int count_);
	void draw(const DrawCommand& drawCommand,GLuint skybox);
	void drawDepth(const DrawCommand& drawCommand);
};