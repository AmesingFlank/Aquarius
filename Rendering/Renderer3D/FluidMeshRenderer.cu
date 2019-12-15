#include "FluidMeshRenderer.cuh"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

FluidMeshRenderer::FluidMeshRenderer(int count_) :count(count_) {
	coordsHost = new float[count * 3 * floatsPerVertex];

	shader = new Shader(Shader::SHADERS_PATH("FluidMeshRenderer_vs.glsl").c_str(), Shader::SHADERS_PATH("FluidMeshRenderer_fs.glsl").c_str(), nullptr);

	vPos_location = glGetAttribLocation(shader->Program, "position");



	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * count * 3 *floatsPerVertex, coordsHost, GL_STATIC_DRAW);
	glEnableVertexAttribArray(vPos_location);
	glVertexAttribPointer(vPos_location, 3, GL_FLOAT, GL_FALSE, sizeof(float) *  floatsPerVertex, 0);

	HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&cudaResourceVBO, VBO, cudaGraphicsMapFlagsNone));

	size_t  size;
	HANDLE_ERROR(cudaGraphicsMapResources(1, &cudaResourceVBO, NULL));
	HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&coordsDevice, &size, cudaResourceVBO));

	glBindVertexArray(0);
	
}

void FluidMeshRenderer::draw(const DrawCommand& drawCommand) {
	shader->Use();
	glBindVertexArray(VAO);

	glm::mat4  model = glm::mat4(1.0);

	GLuint modelLocation = glGetUniformLocation(shader->Program, "model");
	GLuint viewLocation = glGetUniformLocation(shader->Program, "view");
	GLuint projectionLocation = glGetUniformLocation(shader->Program, "projection");

	glUniformMatrix4fv(modelLocation, 1, GL_FALSE, (const GLfloat*)glm::value_ptr(model));
	glUniformMatrix4fv(viewLocation, 1, GL_FALSE, (const GLfloat*)glm::value_ptr(drawCommand.view));
	glUniformMatrix4fv(projectionLocation, 1, GL_FALSE, (const GLfloat*)glm::value_ptr(drawCommand.projection));

	glDrawArrays(GL_TRIANGLES, 0, count*3);
}