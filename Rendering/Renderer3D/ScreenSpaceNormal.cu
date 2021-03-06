#include "ScreenSpaceNormal.cuh"

ScreenSpaceNormal::ScreenSpaceNormal() {
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

	smoothShader = std::make_shared<Shader>(
		Shader::SHADERS_PATH("ScreenSpaceNormal_smooth_vs.glsl").c_str(),
		Shader::SHADERS_PATH("ScreenSpaceNormal_smooth_fs.glsl").c_str()
	);

	normalShader = std::make_shared<Shader>(
		Shader::SHADERS_PATH("ScreenSpaceNormal_normal_vs.glsl").c_str(),
		Shader::SHADERS_PATH("ScreenSpaceNormal_normal_fs.glsl").c_str()
	);


	glGenVertexArrays(1, &quadVAO);
	glGenBuffers(1, &quadVBO);
	glBindVertexArray(quadVAO);
	glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

	GLuint quad_vPos_location, quad_texCoord_location;
	quad_vPos_location = glGetAttribLocation(smoothShader->program, "vPos");
	quad_texCoord_location = glGetAttribLocation(smoothShader->program, "texCoord");

	glEnableVertexAttribArray(quad_vPos_location);
	glVertexAttribPointer(quad_vPos_location, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(quad_texCoord_location);
	glVertexAttribPointer(quad_texCoord_location, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
	printGLError();


}

GLuint ScreenSpaceNormal::generateNormalTexture(std::function<void()> renderDepthFunc,int smoothIterations, int smoothRadius,float sigma_d,float sigma_r, const DrawCommand& drawCommand) {
	renderDepth(renderDepthFunc);
	smoothDepth(smoothIterations,smoothRadius,sigma_d,sigma_r);
	renderNormal(drawCommand);
	return normalTexture;
}

void ScreenSpaceNormal::renderDepth(std::function<void()> renderDepthFunc) {
	glBindFramebuffer(GL_FRAMEBUFFER, FBO);
	glDisable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);
	glClear(GL_DEPTH_BUFFER_BIT);

	GLenum bufs[] = { GL_COLOR_ATTACHMENT0 };
	glDrawBuffers(1, bufs);

	static const float zero[] = { 0, 0, 0, 0 };
	glClearBufferfv(GL_COLOR, 0, zero);

	renderDepthFunc();

	glClear(GL_DEPTH_BUFFER_BIT);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glEnable(GL_BLEND);

	lastDepthTexture = depthTextureA;
}

void ScreenSpaceNormal::smoothDepth(int smoothIterations, int smoothRadius, float sigma_d, float sigma_r) {
	glBindFramebuffer(GL_FRAMEBUFFER, FBO);
	glDisable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);

	smoothShader->use();
	glBindVertexArray(quadVAO);

	smoothShader->setUniform1f("windowWidth", WindowInfo::instance().windowWidth);
	smoothShader->setUniform1f("windowHeight", WindowInfo::instance().windowHeight);

	smoothShader->setUniform1f("sigma_d", sigma_d,true);
	smoothShader->setUniform1f("sigma_r", sigma_r,true);



	int smoothRadiusX = smoothRadius;
	int smoothRadiusY = smoothRadius;


	for (int i = 0; i < smoothIterations; i++) {

		GLuint targetAttachment;
		GLuint nextDepthTexture;

		if (lastDepthTexture == depthTextureA) {
			targetAttachment = GL_COLOR_ATTACHMENT1;
			nextDepthTexture = depthTextureB;
		}
		else {
			targetAttachment = GL_COLOR_ATTACHMENT0;
			nextDepthTexture = depthTextureA;
		}

		glBindFramebuffer(GL_FRAMEBUFFER, FBO);

		smoothShader->use();
		glBindVertexArray(quadVAO);



		GLenum bufs[] = { targetAttachment };
		glDrawBuffers(1, bufs);


		static const float zero[] = { 0,0,0,0 };
		glClearBufferfv(GL_COLOR, 0, zero);

		smoothShader->setUniform1i("smoothRadiusX", smoothRadiusX,true);
		smoothShader->setUniform1i("smoothRadiusY", smoothRadiusY,true);


		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, lastDepthTexture);
		smoothShader->setUniform1i("depthTexture", 0);

		glDrawArrays(GL_TRIANGLES, 0, 6);

		lastDepthTexture = nextDepthTexture;
		std::swap(smoothRadiusX, smoothRadiusY);
	}


	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glEnable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);
}

void ScreenSpaceNormal::renderNormal(const DrawCommand& drawCommand) {
	glBindFramebuffer(GL_FRAMEBUFFER, FBO);
	glDisable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);

	GLenum bufs[] = { GL_COLOR_ATTACHMENT2 };
	glDrawBuffers(1, bufs);

	static const float zero[] = { 0,0,0,0 };
	glClearBufferfv(GL_COLOR, 0, zero);

	normalShader->use();
	glBindVertexArray(quadVAO);


	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, lastDepthTexture);
	normalShader->setUniform1i("depthTexture", 0,true);

	normalShader->setUniform1f("windowWidth", WindowInfo::instance().windowWidth);
	normalShader->setUniform1f("windowHeight", WindowInfo::instance().windowHeight);

	normalShader->setUniform1f("FOV", drawCommand.FOV);


	glDrawArrays(GL_TRIANGLES, 0, 6);


	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glEnable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);
}

ScreenSpaceNormal::~ScreenSpaceNormal() {
	glDeleteBuffers(1, &quadVBO);
	glDeleteVertexArrays(1, &quadVAO);

	glDeleteTextures(1, &depthTextureA);
	glDeleteTextures(1, &depthTextureB);
	glDeleteTextures(1, &depthTextureNDC);
	glDeleteTextures(1, &normalTexture);

	glDeleteFramebuffers(1, &FBO);

}