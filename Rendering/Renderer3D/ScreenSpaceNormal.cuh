#pragma once

#include "../../GpuCommons.h"
#include "../Shader.h"
#include "../WindowInfo.h"

struct ScreenSpaceNormal {
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

	Shader* smoothShader;
	Shader* normalShader;

	ScreenSpaceNormal() {
		glGenTextures(1, &depthTextureNDC);
		glBindTexture(GL_TEXTURE_2D, depthTextureNDC);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, WindowInfo::instance().windowWidth, WindowInfo::instance().windowHeight, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

		glGenTextures(1, &depthTextureA);
		glBindTexture(GL_TEXTURE_2D, depthTextureA);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, WindowInfo::instance().windowWidth, WindowInfo::instance().windowHeight, 0, GL_RED, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

		glGenTextures(1, &depthTextureB);
		glBindTexture(GL_TEXTURE_2D, depthTextureB);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, WindowInfo::instance().windowWidth, WindowInfo::instance().windowHeight, 0, GL_RED, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

		glGenTextures(1, &normalTexture);
		glBindTexture(GL_TEXTURE_2D, normalTexture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, WindowInfo::instance().windowWidth, WindowInfo::instance().windowHeight, 0, GL_RGBA, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);


		glGenFramebuffers(1, &FBO);
		glBindFramebuffer(GL_FRAMEBUFFER, FBO);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTextureNDC, 0);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, depthTextureA, 0);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, depthTextureB, 0);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, normalTexture, 0);

		checkFramebufferComplete();
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		smoothShader = new Shader(
			Shader::SHADERS_PATH("PointSprites_smooth_vs.glsl").c_str(),
			Shader::SHADERS_PATH("PointSprites_smooth_fs.glsl").c_str()
		);

		normalShader = new Shader(
			Shader::SHADERS_PATH("PointSprites_normal_vs.glsl").c_str(),
			Shader::SHADERS_PATH("PointSprites_normal_fs.glsl").c_str()
		);


		glGenVertexArrays(1, &quadVAO);
		glGenBuffers(1, &quadVBO);
		glBindVertexArray(quadVAO);
		glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

		GLuint quad_vPos_location, quad_texCoord_location;
		quad_vPos_location = glGetAttribLocation(smoothShader->Program, "vPos");
		quad_texCoord_location = glGetAttribLocation(smoothShader->Program, "texCoord");

		glEnableVertexAttribArray(quad_vPos_location);
		glVertexAttribPointer(quad_vPos_location, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(quad_texCoord_location);
		glVertexAttribPointer(quad_texCoord_location, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
		printGLError();


	}

};