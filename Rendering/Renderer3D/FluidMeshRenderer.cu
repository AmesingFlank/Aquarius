#include "FluidMeshRenderer.cuh"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#define SCREEN_SPACE_NORMAL 0

FluidMeshRenderer::FluidMeshRenderer(int count_) :count(count_) {
	coordsHost = new float[count * 3 * floatsPerVertex];

	shader = std::make_shared<Shader>(
		Shader::SHADERS_PATH("FluidMeshRenderer_vs.glsl"), 
		Shader::SHADERS_PATH("FluidMeshRenderer_fs.glsl"),
		std::vector<std::string>({ Shader::SHADERS_PATH("RayTraceEnvironment.glsl") })
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

#if SCREEN_SPACE_NORMAL
	screenSpaceNormal = std::make_shared<ScreenSpaceNormal>();
#endif
}



void FluidMeshRenderer::draw(const DrawCommand& drawCommand,  bool multiphase, std::shared_ptr<PointSprites> points, float radius, std::vector<float4> phaseColors) {
#if SCREEN_SPACE_NORMAL
	GLuint screenSpaceNormalTexture = screenSpaceNormal->generateNormalTexture([&]() { drawDepth(drawCommand); }, 6, 5, 6, 1, drawCommand);
#endif

	if (multiphase) {
		
		points->drawPhaseThickness(drawCommand, radius);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		shader->use();

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, points->phaseThicknessTexture);
		shader->setUniform1i("phaseThicknessTexture", 0);


		shader->setUniform1i("usePhaseThicknessTexture", 1);


		for (int i = 0; i < phaseColors.size(); ++i) {
			std::string name = "phaseColors[" + std::to_string(i) + "]";
			float4 color = phaseColors[i];
			shader->setUniform4f(name, color);
		}

		shader->setUniform1i("phaseCount", phaseColors.size());
	}
	else {
		shader->use();
		shader->setUniform1i("usePhaseThicknessTexture", 0);
	}

	
	glm::mat4  model = glm::mat4(1.0);

 
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	
	glBindVertexArray(VAO);

	shader->setUniformMat4("model", model);

#if SCREEN_SPACE_NORMAL
	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, screenSpaceNormalTexture);
	shader->setUniform1i("normalTexture", 2);
#endif
	

	glm::mat4 inverseView = glm::inverse(drawCommand.view);
	shader->setUniformMat4("inverseView", inverseView);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_CUBE_MAP, drawCommand.texSkybox);
	shader->setUniform1i("skybox", 1);

	glActiveTexture(GL_TEXTURE5);
	glBindTexture(GL_TEXTURE_2D, drawCommand.texOxLogo);
	shader->setUniform1i("oxLogo", 5, true);

	shader->setUniformDrawCommand(drawCommand);


	glDrawArrays(GL_TRIANGLES, 0, count);

	

}

void FluidMeshRenderer::drawDepth(const DrawCommand& drawCommand) {
	depthShader->use();
	glBindVertexArray(VAO);

	glm::mat4  model = glm::mat4(1.0);

	depthShader->setUniformMat4("model", model);
	depthShader->setUniformMat4("view", drawCommand.view);
	depthShader->setUniformMat4("projection", drawCommand.projection);

	glDrawArrays(GL_TRIANGLES, 0, count);
}

FluidMeshRenderer::~FluidMeshRenderer() {
	HANDLE_ERROR(cudaGraphicsUnmapResources(1, &cudaResourceVBO));
	HANDLE_ERROR(cudaGraphicsUnregisterResource(cudaResourceVBO));
	glDeleteBuffers(1, &VBO);
	glDeleteVertexArrays(1, &VAO);
}