#include "Container.h"

Container::Container(float size) {
	initEdges();
	initBottom();
	model = glm::scale(model, glm::vec3(size, size, size));
	this->size = size;
	cornellBoxSize = size * 4;
	bigChessBoardSize = size * 100;
}

void Container::drawEdges(const DrawCommand& drawCommand) {
	edgesShader->use();

	edgesShader->setUniformMat4("model", model);
	edgesShader->setUniformMat4("view", drawCommand.view);
	edgesShader->setUniformMat4("projection", drawCommand.projection);

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


void Container::drawFace(const DrawCommand& drawCommand) {
	faceShader->use();

	

	if (drawCommand.environmentMode == EnvironmentMode::CornellBox) {
		float cornellBoxPadding = (cornellBoxSize - size) / 2;


		glm::mat4 cornellScale = glm::scale(model, glm::vec3(cornellBoxSize / size));

		glm::mat4 cornellTranslate(1.0);
		cornellTranslate = glm::translate(cornellTranslate, glm::vec3(-cornellBoxPadding, 0, -cornellBoxPadding));

		glm::mat4 cornellModel = cornellTranslate * cornellScale;

		faceShader->setUniformMat4("model",cornellModel);
	}
	else if (drawCommand.environmentMode == EnvironmentMode::ChessBoard) {
		float padding = (bigChessBoardSize - size) / 2;

		

		glm::mat4 scale = glm::scale(model, glm::vec3(bigChessBoardSize / size));

		glm::mat4 translate(1.0);
		translate = glm::translate(translate, glm::vec3(-padding, 0, -padding));

		glm::mat4 bigChessBoardModel = translate * scale;

		faceShader->setUniformMat4("model", bigChessBoardModel);
	}
	else {
		faceShader->setUniformMat4("model", model);
	}

	
	faceShader->setUniformDrawCommand(drawCommand);


	glBindVertexArray(bottomVAO);

	int drawCount;
	if (drawCommand.environmentMode == EnvironmentMode::CornellBox) {
		drawCount = 36;
	}
	else {
		drawCount = 6;
	}

	glDrawArrays(GL_TRIANGLES, 0, drawCount);
}

void Container::initBottom() {


	faceShader = std::make_shared<Shader>(
		Shader::SHADERS_PATH("Container_face_vs.glsl"),
		Shader::SHADERS_PATH("Container_face_fs.glsl"),
		std::vector<std::string>({ Shader::SHADERS_PATH("RayTraceEnvironment.glsl")})
		);

	glGenVertexArrays(1, &bottomVAO);
	glGenBuffers(1, &bottomVBO);
	glBindVertexArray(bottomVAO);
	glBindBuffer(GL_ARRAY_BUFFER, bottomVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(bottomData), bottomData, GL_STATIC_DRAW);

	GLuint bottomPositionLocation = glGetAttribLocation(faceShader->program, "position");

	glEnableVertexAttribArray(bottomPositionLocation);
	glVertexAttribPointer(bottomPositionLocation, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

	glBindVertexArray(0);
	printGLError();
}

void Container::draw(const DrawCommand& drawCommand) {
	drawFace(drawCommand);
	
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