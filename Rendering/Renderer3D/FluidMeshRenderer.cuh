#pragma once

#include "../../GpuCommons.h"
#include "../Shader.h"
#include "../DrawCommand.h"
#include "ScreenSpaceNormal.cuh"

struct FluidMeshRenderer {
	int count;
	const int floatsPerVertex = 3;
	float* coordsHost;
	float* coordsDevice;
	GLuint VAO, VBO;
	GLint vPos_location;
	cudaGraphicsResource* cudaResourceVBO;
	Shader* shader;
	Shader* depthShader;

	ScreenSpaceNormal screenSpaceNormal;

	FluidMeshRenderer(int count_);
	void draw(const DrawCommand& drawCommand,GLuint skybox);
	void drawDepth(const DrawCommand& drawCommand);
};