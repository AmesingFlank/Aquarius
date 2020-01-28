#pragma once

#include "../../Common/GpuCommons.h"
#include "../Shader.h"
#include "../WindowInfo.h"
#include "../DrawCommand.h"
#include <functional>

class ScreenSpaceNormal {
private:
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
	

	Shader* smoothShader;
	Shader* normalShader;

public:

	GLuint lastDepthTexture;

	GLuint normalTexture;

	ScreenSpaceNormal();

	GLuint generateNormalTexture(std::function<void()> renderDepthFunc,int smoothIterations, int smoothRadius, float sigma_d, float sigma_r,const DrawCommand& drawCommand);


	
private:
	void renderDepth(std::function<void()> renderDepthFunc);
	void smoothDepth(int smoothIterations, int smoothRadius, float sigma_d, float sigma_r);
	void renderNormal(const DrawCommand& drawCommand);

};