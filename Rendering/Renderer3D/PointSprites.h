#pragma once

#include "../../GpuCommons.h"


#include "../Shader.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "../WindowInfo.h"
#include "../DrawCommand.h"

struct PointSprites {
	int count;
	float* positionsHost;
	GLuint pointsVAO, pointsVBO;
	GLint points_vPos_location; // used by multiple shaders. location specified as common value in all shaders

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
	// view space
	GLuint depthTextureA;
	GLuint depthTextureB;
	GLuint lastDepthTexture;

	GLuint normalTexture;
	GLuint thicknessTexture;

	Shader* depthShader;
	Shader* renderShader;
	Shader* normalShader;
	Shader* thicknessShader;
	Shader* smoothShader;

	glm::mat4  model = glm::mat4(1.0);


	void initScreenSpaceRenderer();
	

	void renderDepth(const DrawCommand& drawCommand, float radius);
	void smoothDepth(const DrawCommand& drawCommand, int iterations, int smoothRadius, float sigma_d, float sigma_r);

	void renderNormal(const DrawCommand& drawCommand);
	void renderThickness(const DrawCommand& drawCommand, float radius);
	void renderFinal(const DrawCommand& drawCommand, int skybox);

	PointSprites(int count_);


	void draw(const DrawCommand& drawCommand, float radius, int skybox);

	void drawSimple(const DrawCommand& drawCommand, float radius);
};