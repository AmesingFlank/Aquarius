#pragma once

#include "../../Common/GpuCommons.h"


#include "../Shader.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "../WindowInfo.h"
#include "../DrawCommand.h"
#include "ScreenSpaceNormal.cuh"
#include <memory>


struct PointSprites {
	int count;
	float* pointsVBO_host;
	GLuint pointsVAO, pointsVBO;

	int stride = 7;
	

	cudaGraphicsResource* cudaResourceVBO;
	float* positionsDevice;

	

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

	GLuint phaseThicknessTexture;

	std::shared_ptr<Shader> simpleShader;
	std::shared_ptr<Shader> depthShader;
	std::shared_ptr<Shader> screenShader;
	std::shared_ptr<Shader> thicknessShader;
	std::shared_ptr<Shader> phaseThicknessShader;


	ScreenSpaceNormal screenSpaceNormal;

	glm::mat4  model = glm::mat4(1.0);


	void initRenderer();
	

	void drawDepth(const DrawCommand& drawCommand, float radius);

	void drawThickness(const DrawCommand& drawCommand, float radius);
	void drawScreen(const DrawCommand& drawCommand, int skybox,GLuint normalTexture,GLuint depthTexture);

	PointSprites(int count_);


	void draw(const DrawCommand& drawCommand, float radius, int skybox);

	void drawSimple(const DrawCommand& drawCommand, float radius);
	void drawPhaseThickness(const DrawCommand& drawCommand, float radius);


	void prepareShader(std::shared_ptr<Shader> shader, const DrawCommand& drawCommand, float radius);

	~PointSprites();
	
};