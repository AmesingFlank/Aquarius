#include "FluidMeshRenderer.cuh"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

FluidMeshRenderer::FluidMeshRenderer(int count_) :count(count_) {
	coordsHost = new float[count * 3 * floatsPerVertex];

	shader = std::make_shared<Shader>(
		Shader::SHADERS_PATH("FluidMeshRenderer_vs.glsl"), 
		Shader::SHADERS_PATH("FluidMeshRenderer_fs.glsl")
	);

	depthShader = std::make_shared<Shader>(
		Shader::SHADERS_PATH("FluidMeshRenderer_vs.glsl"),
		Shader::SHADERS_PATH("FluidMeshRenderer_depth_fs.glsl")
	);

	positionLocation = glGetAttribLocation(shader->program, "position");
	normalLocation = glGetAttribLocation(shader->program, "normal");


	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * count * 3 *floatsPerVertex, coordsHost, GL_STATIC_DRAW);

	glEnableVertexAttribArray(positionLocation);
	glVertexAttribPointer(positionLocation, 3, GL_FLOAT, GL_FALSE, sizeof(float) *  floatsPerVertex, 0);

	glEnableVertexAttribArray(normalLocation);
	glVertexAttribPointer(normalLocation, 3, GL_FLOAT, GL_FALSE, sizeof(float) * floatsPerVertex, 
		(void*) (3*sizeof(float)));

	HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&cudaResourceVBO, VBO, cudaGraphicsMapFlagsNone));

	size_t  size;
	HANDLE_ERROR(cudaGraphicsMapResources(1, &cudaResourceVBO, NULL));
	HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&coordsDevice, &size, cudaResourceVBO));

	glBindVertexArray(0);
	
	delete[] coordsHost;
}

void FluidMeshRenderer::drawWithInk(const DrawCommand& drawCommand, GLuint skybox, PointSprites& points, float radius, std::vector<float4> phaseColors) {
	

	points.drawPhaseThickness(drawCommand, radius);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	shader->use();

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, points.phaseThicknessTexture);
	GLuint phaseThicknessTextureLocation = glGetUniformLocation(shader->program, "phaseThicknessTexture");
	glUniform1i(phaseThicknessTextureLocation, 0);

	GLuint usePhaseThicknessTextureLocation = glGetUniformLocation(shader->program, "usePhaseThicknessTexture");
	glUniform1i(usePhaseThicknessTextureLocation, 1);


	for (int i = 0; i < phaseColors.size(); ++i) {
		std::string name = "phaseColors[" + std::to_string(i) + "]";
		GLuint location = glGetUniformLocation(shader->program, name.c_str());
		float4 color = phaseColors[i];
		glUniform4f(location, color.x, color.y, color.z, color.w);
	}

	GLuint phaseCountLocation = glGetUniformLocation(shader->program,"phaseCount");
	glUniform1i(phaseCountLocation,phaseColors.size());


	draw(drawCommand, skybox); 


	return;

}

void FluidMeshRenderer::draw(const DrawCommand& drawCommand, GLuint skybox) {
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	
	glm::mat4  model = glm::mat4(1.0);

	
	//GLuint thicknessTexture;
	//points.drawThickness(drawCommand, radius);
	
 
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	shader->use();
	glBindVertexArray(VAO);

	GLuint modelLocation = glGetUniformLocation(shader->program, "model");
	GLuint viewLocation = glGetUniformLocation(shader->program, "view");
	GLuint projectionLocation = glGetUniformLocation(shader->program, "projection");

	glUniformMatrix4fv(modelLocation, 1, GL_FALSE, (const GLfloat*)glm::value_ptr(model));
	glUniformMatrix4fv(viewLocation, 1, GL_FALSE, (const GLfloat*)glm::value_ptr(drawCommand.view));
	glUniformMatrix4fv(projectionLocation, 1, GL_FALSE, (const GLfloat*)glm::value_ptr(drawCommand.projection));

	/*
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, screenSpaceNormalTexture);
	GLuint normalTextureLocation = glGetUniformLocation(shader->program, "normalTexture");
	glUniform1i(normalTextureLocation, 0);
	*/

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_CUBE_MAP, skybox);
	GLuint skyboxLocation = glGetUniformLocation(shader->program, "skybox");
	glUniform1i(skyboxLocation, 1);

	glm::mat4 inverseView = glm::inverse(drawCommand.view);
	GLuint inverseViewLocation = glGetUniformLocation(shader->program, "inverseView");
	glUniformMatrix4fv(inverseViewLocation, 1, GL_FALSE, (const GLfloat*)glm::value_ptr(inverseView));

	glm::vec3 cameraPos = drawCommand.cameraPosition;
	GLuint cameraPositionLocation = glGetUniformLocation(shader->program, "cameraPosition");
	glUniform3f(cameraPositionLocation, cameraPos.x, cameraPos.y, cameraPos.z);


	glDrawArrays(GL_TRIANGLES, 0, count);

	

}

void FluidMeshRenderer::drawDepth(const DrawCommand& drawCommand) {
	depthShader->use();
	glBindVertexArray(VAO);

	glm::mat4  model = glm::mat4(1.0);

	GLuint modelLocation = glGetUniformLocation(depthShader->program, "model");
	GLuint viewLocation = glGetUniformLocation(depthShader->program, "view");
	GLuint projectionLocation = glGetUniformLocation(depthShader->program, "projection");

	glUniformMatrix4fv(modelLocation, 1, GL_FALSE, (const GLfloat*)glm::value_ptr(model));
	glUniformMatrix4fv(viewLocation, 1, GL_FALSE, (const GLfloat*)glm::value_ptr(drawCommand.view));
	glUniformMatrix4fv(projectionLocation, 1, GL_FALSE, (const GLfloat*)glm::value_ptr(drawCommand.projection));

	glDrawArrays(GL_TRIANGLES, 0, count);
}

FluidMeshRenderer::~FluidMeshRenderer() {
	HANDLE_ERROR(cudaGraphicsUnmapResources(1, &cudaResourceVBO));
	HANDLE_ERROR(cudaGraphicsUnregisterResource(cudaResourceVBO));
	glDeleteBuffers(1, &VBO);
	glDeleteVertexArrays(1, &VAO);
}