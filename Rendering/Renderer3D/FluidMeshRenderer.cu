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

	//screenSpaceNormal = std::make_shared<ScreenSpaceNormal>();
}

void FluidMeshRenderer::drawWithInk(const DrawCommand& drawCommand, GLuint skybox, PointSprites& points, float radius, std::vector<float4> phaseColors) {
	

	//GLuint screenSpaceNormalTexture = screenSpaceNormal->generateNormalTexture([&]() { drawDepth(drawCommand); }, 6, 5, 6, 0.1, drawCommand);

	points.drawPhaseThickness(drawCommand, radius);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	shader->use();

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, points.phaseThicknessTexture);
	shader->setUniform1i("phaseThicknessTexture", 0);


	shader->setUniform1i("usePhaseThicknessTexture", 1);


	for (int i = 0; i < phaseColors.size(); ++i) {
		std::string name = "phaseColors[" + std::to_string(i) + "]";
		float4 color = phaseColors[i];
		shader->setUniform4f(name, color);
	}

	shader->setUniform1i("phaseCount", phaseColors.size());


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

	shader->setUniformMat4("model", model);
	shader->setUniformMat4("view", drawCommand.view);
	shader->setUniformMat4("projection", drawCommand.projection);

	
	//glActiveTexture(GL_TEXTURE0);
	//glBindTexture(GL_TEXTURE_2D, screenSpaceNormalTexture);
	//shader->setUniform1i("normalTexture", 0);
	

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_CUBE_MAP, skybox);
	shader->setUniform1i("skybox", 1);

	glm::mat4 inverseView = glm::inverse(drawCommand.view);
	shader->setUniformMat4("inverseView", inverseView);

	glm::vec3 cameraPos = drawCommand.cameraPosition;
	shader->setUniform3f("cameraPosition", drawCommand.cameraPosition);


	glDrawArrays(GL_TRIANGLES, 0, count);

	

}

void FluidMeshRenderer::drawDepth(const DrawCommand& drawCommand) {
	depthShader->use();
	glBindVertexArray(VAO);

	glm::mat4  model = glm::mat4(1.0);

	shader->setUniformMat4("model", model);
	shader->setUniformMat4("view", drawCommand.view);
	shader->setUniformMat4("projection", drawCommand.projection);

	glDrawArrays(GL_TRIANGLES, 0, count);
}

FluidMeshRenderer::~FluidMeshRenderer() {
	HANDLE_ERROR(cudaGraphicsUnmapResources(1, &cudaResourceVBO));
	HANDLE_ERROR(cudaGraphicsUnregisterResource(cudaResourceVBO));
	glDeleteBuffers(1, &VBO);
	glDeleteVertexArrays(1, &VAO);
}