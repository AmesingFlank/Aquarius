#pragma once

#include "../../Common/GpuCommons.h"
#include "../Shader.h"
#include "../WindowInfo.h"
#include "../DrawCommand.h"
#include <functional>
#include <memory>


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
	

	std::shared_ptr<Shader> smoothShader;
	std::shared_ptr<Shader> normalShader;

public:

	GLuint lastDepthTexture;

	GLuint normalTexture;

	ScreenSpaceNormal();

	GLuint generateNormalTexture(std::function<void()> renderDepthFunc,int smoothIterations, int smoothRadius, float sigma_d, float sigma_r,const DrawCommand& drawCommand);
	~ScreenSpaceNormal();

	
private:
	void renderDepth(std::function<void()> renderDepthFunc);
	void smoothDepth(int smoothIterations, int smoothRadius, float sigma_d, float sigma_r);
	void renderNormal(const DrawCommand& drawCommand);

	

};