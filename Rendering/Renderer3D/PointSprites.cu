#include "PointSprites.h"
#include "../DrawCommand.h"

void PointSprites::draw(const DrawCommand& drawCommand, float radius, int skybox) {
	//return;
	if (drawCommand.renderMode == RenderMode::Particles) {
		drawSimple(drawCommand, radius); return;
	}
	
	GLuint normalTexture = screenSpaceNormal.generateNormalTexture([&]() 
		{
			renderDepth(drawCommand,radius);
		},
		6, 5, 6, 0.1, drawCommand);
	GLuint depthTexture = screenSpaceNormal.lastDepthTexture;

	renderThickness(drawCommand, radius);

	renderFinal(drawCommand, skybox,normalTexture,depthTexture);
	printGLError();
}

void PointSprites::initRenderer() {

	pointsVBO_host = new float[count * stride];

	pointsShader = new Shader(Shader::SHADERS_PATH("PointSprites_points_vs.glsl").c_str(), Shader::SHADERS_PATH("PointSprites_points_fs.glsl").c_str(), nullptr);

	inkShader = new Shader(Shader::SHADERS_PATH("PointSprites_points_vs.glsl").c_str(), Shader::SHADERS_PATH("PointSprites_ink_fs.glsl").c_str(), nullptr);


	// used by multiple shaders. location specified as common value in all shader code
	GLint pointsPositionLocation = glGetAttribLocation(pointsShader->Program, "position");
	GLint pointsColorLocation = glGetAttribLocation(pointsShader->Program, "color");


	glGenVertexArrays(1, &pointsVAO);
	glGenBuffers(1, &pointsVBO);
	glBindVertexArray(pointsVAO);
	glBindBuffer(GL_ARRAY_BUFFER, pointsVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * count * 7, pointsVBO_host, GL_STATIC_DRAW);

	glEnableVertexAttribArray(pointsPositionLocation);
	glVertexAttribPointer(pointsPositionLocation, 3, GL_FLOAT, GL_FALSE, sizeof(float) * stride, 0);

	glEnableVertexAttribArray(pointsColorLocation);
	glVertexAttribPointer(pointsColorLocation, 4, GL_FLOAT, GL_FALSE, sizeof(float) * stride, (void*)(sizeof(float) * 3));

	HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&cudaResourceVBO, pointsVBO, cudaGraphicsMapFlagsNone));

	size_t  size;
	HANDLE_ERROR(cudaGraphicsMapResources(1, &cudaResourceVBO, NULL));
	HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&positionsDevice, &size, cudaResourceVBO));

	glBindVertexArray(0);





	glGenTextures(1, &depthTextureNDC);
	glBindTexture(GL_TEXTURE_2D, depthTextureNDC);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, WindowInfo::instance().windowWidth, WindowInfo::instance().windowHeight, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glGenTextures(1, &thicknessTexture);
	glBindTexture(GL_TEXTURE_2D, thicknessTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, WindowInfo::instance().windowWidth, WindowInfo::instance().windowHeight, 0, GL_RGBA, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glGenFramebuffers(1, &FBO);
	glBindFramebuffer(GL_FRAMEBUFFER, FBO);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTextureNDC, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D, thicknessTexture, 0);


	checkFramebufferComplete();
	glBindFramebuffer(GL_FRAMEBUFFER, 0);


	depthShader = new Shader(
		Shader::SHADERS_PATH("PointSprites_depth_vs.glsl").c_str(),
		Shader::SHADERS_PATH("PointSprites_depth_fs.glsl").c_str()
	);
	renderShader = new Shader(
		Shader::SHADERS_PATH("PointSprites_render_vs.glsl").c_str(),
		Shader::SHADERS_PATH("PointSprites_render_fs.glsl").c_str()
	);
	

	thicknessShader = new Shader(
		Shader::SHADERS_PATH("PointSprites_thickness_vs.glsl").c_str(),
		Shader::SHADERS_PATH("PointSprites_thickness_fs.glsl").c_str()
	);

	glGenVertexArrays(1, &quadVAO);
	glGenBuffers(1, &quadVBO);
	glBindVertexArray(quadVAO);
	glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

	GLuint quad_vPos_location, quad_texCoord_location;
	quad_vPos_location = glGetAttribLocation(renderShader->Program, "vPos");
	quad_texCoord_location = glGetAttribLocation(renderShader->Program, "texCoord");

	glEnableVertexAttribArray(quad_vPos_location);
	glVertexAttribPointer(quad_vPos_location, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(quad_texCoord_location);
	glVertexAttribPointer(quad_texCoord_location, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
	printGLError();

}



void PointSprites::renderDepth(const DrawCommand& drawCommand, float radius) {
	glm::mat4 view = drawCommand.view;
	glm::mat4 projection = drawCommand.projection;
	glm::vec3 cameraPos = drawCommand.cameraPosition;

	depthShader->Use();


	glUniformMatrix4fv(glGetUniformLocation(depthShader->Program, "model")
		, 1, GL_FALSE, (const GLfloat*)glm::value_ptr(model));
	glUniformMatrix4fv(glGetUniformLocation(depthShader->Program, "view")
		, 1, GL_FALSE, (const GLfloat*)glm::value_ptr(view));
	glUniformMatrix4fv(glGetUniformLocation(depthShader->Program, "projection")
		, 1, GL_FALSE, (const GLfloat*)glm::value_ptr(projection));

	glUniform1f(glGetUniformLocation(depthShader->Program, "windowWidth"), drawCommand.windowWidth);
	glUniform1f(glGetUniformLocation(depthShader->Program, "windowHeight"), drawCommand.windowHeight);

	glUniform1f(glGetUniformLocation(depthShader->Program, "radius"), radius);

	glUniform3f(glGetUniformLocation(depthShader->Program, "cameraPosition"), cameraPos.x, cameraPos.y, cameraPos.z);

	glBindVertexArray(pointsVAO);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glPointParameteri(GL_POINT_SPRITE_COORD_ORIGIN, GL_LOWER_LEFT);

	glDrawArrays(GL_POINTS, 0, count);

}


void PointSprites::renderThickness(const DrawCommand& drawCommand, float radius) {
	glBindFramebuffer(GL_FRAMEBUFFER, FBO);
	glEnable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);

	glBlendFuncSeparate(GL_ONE, GL_ONE, GL_ONE, GL_ONE);

	glClear(GL_DEPTH_BUFFER_BIT);

	glm::mat4 view = drawCommand.view;
	glm::mat4 projection = drawCommand.projection;
	glm::vec3 cameraPos = drawCommand.cameraPosition;

	thicknessShader->Use();

	GLenum bufs[] = { GL_COLOR_ATTACHMENT3 };
	glDrawBuffers(1, bufs);

	static const float zero[] = { 0, 0, 0, 0 };
	glClearBufferfv(GL_COLOR, 0, zero);

	glUniformMatrix4fv(glGetUniformLocation(thicknessShader->Program, "model")
		, 1, GL_FALSE, (const GLfloat*)glm::value_ptr(model));
	glUniformMatrix4fv(glGetUniformLocation(thicknessShader->Program, "view")
		, 1, GL_FALSE, (const GLfloat*)glm::value_ptr(view));
	glUniformMatrix4fv(glGetUniformLocation(thicknessShader->Program, "projection")
		, 1, GL_FALSE, (const GLfloat*)glm::value_ptr(projection));

	glUniform1f(glGetUniformLocation(thicknessShader->Program, "windowWidth"), drawCommand.windowWidth);
	glUniform1f(glGetUniformLocation(thicknessShader->Program, "windowHeight"), drawCommand.windowHeight);

	glUniform1f(glGetUniformLocation(thicknessShader->Program, "radius"), radius);

	glUniform3f(glGetUniformLocation(thicknessShader->Program, "cameraPosition"), cameraPos.x, cameraPos.y, cameraPos.z);

	glBindVertexArray(pointsVAO);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glPointParameteri(GL_POINT_SPRITE_COORD_ORIGIN, GL_LOWER_LEFT);

	glDrawArrays(GL_POINTS, 0, count);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glEnable(GL_DEPTH_TEST);

	glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	//glBlendFuncSeparate(GL_ONE, GL_ZERO, GL_ONE, GL_ZERO);

}

void PointSprites::renderFinal(const DrawCommand& drawCommand, int skybox,GLuint normalTexture,GLuint depthTexture) {
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);

	renderShader->Use();
	glBindVertexArray(quadVAO);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, depthTexture);
	GLuint depthTextureLocation = glGetUniformLocation(renderShader->Program, "depthTexture");
	glUniform1i(depthTextureLocation, 0);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, normalTexture);
	GLuint normalTextureLocation = glGetUniformLocation(renderShader->Program, "normalTexture");
	glUniform1i(normalTextureLocation, 1);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, thicknessTexture);
	GLuint thicknessTextureLocation = glGetUniformLocation(renderShader->Program, "thicknessTexture");
	glUniform1i(thicknessTextureLocation, 2);

	glActiveTexture(GL_TEXTURE3);
	glBindTexture(GL_TEXTURE_CUBE_MAP, skybox);
	GLuint skyboxLocation = glGetUniformLocation(renderShader->Program, "skybox");
	glUniform1i(skyboxLocation, 3);


	GLuint projectionLocation = glGetUniformLocation(renderShader->Program, "projection");
	glUniformMatrix4fv(projectionLocation, 1, GL_FALSE, (const GLfloat*)glm::value_ptr(drawCommand.projection));

	glm::mat4 inverseView = glm::inverse(drawCommand.view);
	GLuint inverseViewLocation = glGetUniformLocation(renderShader->Program, "inverseView");
	glUniformMatrix4fv(inverseViewLocation, 1, GL_FALSE, (const GLfloat*)glm::value_ptr(inverseView));

	GLuint windowWidthLocation = glGetUniformLocation(renderShader->Program, "windowWidth");
	glUniform1f(windowWidthLocation, drawCommand.windowWidth);

	GLuint windowHeightLocation = glGetUniformLocation(renderShader->Program, "windowHeight");
	glUniform1f(windowHeightLocation, drawCommand.windowHeight);

	GLuint zoomLocation = glGetUniformLocation(renderShader->Program, "zoom");
	glUniform1f(zoomLocation, drawCommand.zoom);


	glm::vec3 cameraPos = drawCommand.cameraPosition;
	GLuint cameraPositionLocation = glGetUniformLocation(renderShader->Program, "cameraPosition");
	glUniform3f(cameraPositionLocation, cameraPos.x, cameraPos.y, cameraPos.z);

	glDrawArrays(GL_TRIANGLES, 0, 6);

}

PointSprites::PointSprites(int count_) :count(count_) {

	

	initRenderer();

};




void PointSprites::drawSimple(const DrawCommand& drawCommand, float radius) {

	glEnable(GL_BLEND);
	//glDisable(GL_DEPTH_TEST);
	glEnable(GL_DEPTH_TEST);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glBlendEquation(GL_FUNC_ADD);

	glm::mat4 view = drawCommand.view;
	glm::mat4 projection = drawCommand.projection;
	glm::vec3 cameraPos = drawCommand.cameraPosition;

	pointsShader->Use();
	GLuint modelLocation = glGetUniformLocation(pointsShader->Program, "model");
	GLuint viewLocation = glGetUniformLocation(pointsShader->Program, "view");
	GLuint projectionLocation = glGetUniformLocation(pointsShader->Program, "projection");

	glUniformMatrix4fv(modelLocation, 1, GL_FALSE, (const GLfloat*)glm::value_ptr(model));
	glUniformMatrix4fv(viewLocation, 1, GL_FALSE, (const GLfloat*)glm::value_ptr(view));
	glUniformMatrix4fv(projectionLocation, 1, GL_FALSE, (const GLfloat*)glm::value_ptr(projection));

	glUniform1f(glGetUniformLocation(pointsShader->Program, "windowWidth"), drawCommand.windowWidth);
	glUniform1f(glGetUniformLocation(pointsShader->Program, "windowHeight"), drawCommand.windowHeight);

	glUniform1f(glGetUniformLocation(pointsShader->Program, "radius"), radius);

	glUniform3f(glGetUniformLocation(pointsShader->Program, "cameraPosition"), cameraPos.x, cameraPos.y, cameraPos.z);


	glBindVertexArray(pointsVAO);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glPointParameteri(GL_POINT_SPRITE_COORD_ORIGIN, GL_LOWER_LEFT);
	//glPointSize(50);
	glDrawArrays(GL_POINTS, 0, count);
	glEnable(GL_DEPTH_TEST);


}


void PointSprites::drawInk(const DrawCommand& drawCommand, float radius) {

	glEnable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glBlendEquation(GL_FUNC_ADD);

	glm::mat4 view = drawCommand.view;
	glm::mat4 projection = drawCommand.projection;
	glm::vec3 cameraPos = drawCommand.cameraPosition;

	inkShader->Use();
	GLuint modelLocation = glGetUniformLocation(inkShader->Program, "model");
	GLuint viewLocation = glGetUniformLocation(inkShader->Program, "view");
	GLuint projectionLocation = glGetUniformLocation(inkShader->Program, "projection");

	glUniformMatrix4fv(modelLocation, 1, GL_FALSE, (const GLfloat*)glm::value_ptr(model));
	glUniformMatrix4fv(viewLocation, 1, GL_FALSE, (const GLfloat*)glm::value_ptr(view));
	glUniformMatrix4fv(projectionLocation, 1, GL_FALSE, (const GLfloat*)glm::value_ptr(projection));

	glUniform1f(glGetUniformLocation(inkShader->Program, "windowWidth"), drawCommand.windowWidth);
	glUniform1f(glGetUniformLocation(inkShader->Program, "windowHeight"), drawCommand.windowHeight);

	glUniform1f(glGetUniformLocation(inkShader->Program, "radius"), radius);

	glUniform3f(glGetUniformLocation(inkShader->Program, "cameraPosition"), cameraPos.x, cameraPos.y, cameraPos.z);


	glBindVertexArray(pointsVAO);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glPointParameteri(GL_POINT_SPRITE_COORD_ORIGIN, GL_LOWER_LEFT);
	//glPointSize(50);
	glDrawArrays(GL_POINTS, 0, count);
	glEnable(GL_DEPTH_TEST);


}

