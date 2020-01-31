#pragma once

#include "../../Common/GpuCommons.h"


#include "../Shader.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "../WindowInfo.h"
#include "../DrawCommand.h"
#include "ScreenSpaceNormal.cuh"


struct PointSprites {
	int count;
	float* pointsVBO_host;
	GLuint pointsVAO, pointsVBO;
	

	cudaGraphicsResource* cudaResourceVBO;
	float* positionsDevice;

	Shader* basicShader;

	float quadVertices[24] = { 
		// positions   // texCoords
		-1.0f,  1.0f,  0.0f, 1.0f,
		-1.0f, -1.0f,  0.0f, 0.0f,
		1.0f, -1.0f,  1.0f, 0.0f,

		-1.0f,  1.0f,  0.0f, 1.0f,
		1.0f, -1.0f,  1.0f, 0.0f,
		1.0f,  1.0f,  1.0f, 1.0f
	};

	GLuint quadVBO, quadVAO;

	GLuint FBO;

	GLuint depthTextureNDC;

	GLuint thicknessTexture;

	Shader* depthShader;
	Shader* renderShader;

	Shader* thicknessShader;


	ScreenSpaceNormal screenSpaceNormal;

	glm::mat4  model = glm::mat4(1.0);


	void initScreenSpaceRenderer();
	

	void renderDepth(const DrawCommand& drawCommand, float radius);

	void renderThickness(const DrawCommand& drawCommand, float radius);
	void renderFinal(const DrawCommand& drawCommand, int skybox,GLuint normalTexture,GLuint depthTexture);

	PointSprites(int count_);


	void draw(const DrawCommand& drawCommand, float radius, int skybox);

	void drawSimple(const DrawCommand& drawCommand, float radius);
};