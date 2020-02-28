#include "Container.h"

Container::Container(float size) {
	initEdges();
	initBottom();
	model = glm::scale(model, glm::vec3(size, size, size));
	this->size = size;
}

void Container::drawEdges(const DrawCommand& drawCommand) {
	edgesShader->use();

	GLuint modelLocation = glGetUniformLocation(edgesShader->program, "model");
	GLuint viewLocation = glGetUniformLocation(edgesShader->program, "view");
	GLuint projectionLocation = glGetUniformLocation(edgesShader->program, "projection");

	glUniformMatrix4fv(modelLocation, 1, GL_FALSE, (const GLfloat*)glm::value_ptr(model));
	glUniformMatrix4fv(viewLocation, 1, GL_FALSE, (const GLfloat*)glm::value_ptr(drawCommand.view));
	glUniformMatrix4fv(projectionLocation, 1, GL_FALSE, (const GLfloat*)glm::value_ptr(drawCommand.projection));

	glBindVertexArray(edgesVAO);

	glDrawArrays(GL_LINES, 0, 24);
}

void Container::initEdges() {

	edgesShader = std::make_shared<Shader>(
		Shader::SHADERS_PATH("Container_edges_vs.glsl"),
		Shader::SHADERS_PATH("Container_edges_fs.glsl")
	);

	glGenVertexArrays(1, &edgesVAO);
	glGenBuffers(1, &edgesVBO);
	glBindVertexArray(edgesVAO);
	glBindBuffer(GL_ARRAY_BUFFER, edgesVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(edgesData),edgesData, GL_STATIC_DRAW);

	GLuint edgesPositionLocation = glGetAttribLocation(edgesShader->program,"position");

	glEnableVertexAttribArray(edgesPositionLocation);
	glVertexAttribPointer(edgesPositionLocation, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

	glBindVertexArray(0);
	printGLError();
}


void Container::drawBottom(const DrawCommand& drawCommand) {
	bottomShader->use();

	float extension = 1;

	glm::mat4 bottomScale = glm::scale(model, glm::vec3(2 + extension));

	glm::mat4 bottomTranslate(1.0);
	bottomTranslate = glm::translate(bottomTranslate, glm::vec3(-size, 0, -size));

	glm::mat4 bottomModel = bottomTranslate * bottomScale;



	GLuint modelLocation = glGetUniformLocation(bottomShader->program, "model");
	GLuint viewLocation = glGetUniformLocation(bottomShader->program, "view");
	GLuint projectionLocation = glGetUniformLocation(bottomShader->program, "projection");

	glUniformMatrix4fv(modelLocation, 1, GL_FALSE, (const GLfloat*)glm::value_ptr(bottomModel));
	glUniformMatrix4fv(viewLocation, 1, GL_FALSE, (const GLfloat*)glm::value_ptr(drawCommand.view));
	glUniformMatrix4fv(projectionLocation, 1, GL_FALSE, (const GLfloat*)glm::value_ptr(drawCommand.projection));

	GLuint cameraPosLocation = glGetUniformLocation(bottomShader->program, "cameraPos");
	glUniform3f(cameraPosLocation, drawCommand.cameraPosition.x, drawCommand.cameraPosition.y, drawCommand.cameraPosition.z);

	GLuint lightPosLocation = glGetUniformLocation(bottomShader->program, "lightPos");
	glUniform3f(lightPosLocation, drawCommand.lightPos.x, drawCommand.lightPos.y, drawCommand.lightPos.z);

	GLuint boxSizeLocation = glGetUniformLocation(bottomShader->program, "boxSize");
	glUniform1f(boxSizeLocation, size);

	glBindVertexArray(bottomVAO);

	glDrawArrays(GL_TRIANGLES, 0, 36);
}

void Container::initBottom() {


	bottomShader = std::make_shared<Shader>(
		Shader::SHADERS_PATH("Container_bottom_vs.glsl"),
		Shader::SHADERS_PATH("Container_bottom_fs.glsl"),
		std::vector<std::string>({ Shader::SHADERS_PATH("RayTraceEnvironment.glsl")})
		);

	glGenVertexArrays(1, &bottomVAO);
	glGenBuffers(1, &bottomVBO);
	glBindVertexArray(bottomVAO);
	glBindBuffer(GL_ARRAY_BUFFER, bottomVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(bottomData), bottomData, GL_STATIC_DRAW);

	GLuint bottomPositionLocation = glGetAttribLocation(bottomShader->program, "position");

	glEnableVertexAttribArray(bottomPositionLocation);
	glVertexAttribPointer(bottomPositionLocation, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

	glBindVertexArray(0);
	printGLError();
}

void Container::draw(const DrawCommand& drawCommand) {
	//drawBottom(drawCommand);
	
	glDisable(GL_DEPTH_TEST);
	drawEdges(drawCommand);
	glEnable(GL_DEPTH_TEST);

}

Container::~Container() {
	glDeleteBuffers(1, &bottomVBO);
	glDeleteBuffers(1, &edgesVBO);

	glDeleteVertexArrays(1, &bottomVAO);
	glDeleteVertexArrays(1, &bottomVBO);

}