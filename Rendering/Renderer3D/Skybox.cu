#include "Skybox.h"

Skybox::Skybox(const std::string& path, const std::string& extension) {

	std::string vsPath = Shader::SHADERS_PATH("Skybox_vs.glsl");
	std::string fsPath = Shader::SHADERS_PATH("Skybox_fs.glsl");

	shader = std::make_shared<Shader>(vsPath, fsPath);


	vPos_location = glGetAttribLocation(shader->program, "position");
	model_location = glGetUniformLocation(shader->program, "model");
	view_location = glGetUniformLocation(shader->program, "view");
	projection_location = glGetUniformLocation(shader->program, "projection");

	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glEnableVertexAttribArray(vPos_location);
	glVertexAttribPointer(vPos_location, 3, GL_FLOAT, GL_FALSE,
		sizeof(float) * 3, (void*)(0));

	glBindVertexArray(0);


	std::vector<std::string> skyBoxFaces = { "posx",
								 "negx",
								 "posy",
								 "negy",
								 "posz",
								 "negz", };
	glGenTextures(1, &texSkyBox);
	glBindTexture(GL_TEXTURE_CUBE_MAP, texSkyBox);

	for (int i = 0; i < skyBoxFaces.size(); ++i) {
		int width, height;
		const std::string file = path + skyBoxFaces[i] + extension;
		unsigned const char* image = stbi_load(file.c_str(), &width, &height, 0, STBI_rgb);
		if (!image) {
			std::cerr << "read image failed: " << file << std::endl;
		}
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image);
	}

	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

}

void Skybox::draw(const DrawCommand& drawCommand) {

	glDepthMask(GL_FALSE);
	glUseProgram(shader->program);
	glBindVertexArray(VAO);
	glBindTexture(GL_TEXTURE_CUBE_MAP, texSkyBox);

	glm::mat4 newView = glm::mat4(glm::mat3(drawCommand.view));
	glUniformMatrix4fv(model_location, 1, GL_FALSE, (const GLfloat*)glm::value_ptr(model));
	glUniformMatrix4fv(view_location, 1, GL_FALSE, (const GLfloat*)glm::value_ptr(newView));
	glUniformMatrix4fv(projection_location, 1, GL_FALSE, (const GLfloat*)glm::value_ptr(drawCommand.projection));

	glDrawArrays(GL_TRIANGLES, 0, 36);
	glBindVertexArray(0);
	glDepthMask(GL_TRUE);
}

Skybox::~Skybox() {
	glDeleteBuffers(1, &VBO);
	glDeleteVertexArrays(1, &VAO);

	glDeleteTextures(1, &texSkyBox);
}