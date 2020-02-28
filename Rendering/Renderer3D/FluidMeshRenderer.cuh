#pragma once

#include "../../Common/GpuCommons.h"
#include "../Shader.h"
#include "../DrawCommand.h"
#include "ScreenSpaceNormal.cuh"
#include "PointSprites.h"
#include <vector>
#include <memory>


struct FluidMeshRenderer {
	int count;
	const int floatsPerVertex = 6;
	float* coordsHost;
	float* coordsDevice;
	GLuint VAO, VBO;
	GLint positionLocation;
	GLint normalLocation;
	cudaGraphicsResource* cudaResourceVBO;

	std::shared_ptr<Shader> shader;
	std::shared_ptr<Shader> depthShader;

	ScreenSpaceNormal screenSpaceNormal;

	FluidMeshRenderer(int count_);
	void draw(const DrawCommand& drawCommand, GLuint skybox);
	void drawWithInk(const DrawCommand& drawCommand, GLuint skybox, PointSprites& points, float radius, std::vector<float4> phaseColors);
	void drawDepth(const DrawCommand& drawCommand);

	~FluidMeshRenderer();
};