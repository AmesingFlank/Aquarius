#pragma once
#include "../model.h"

struct Container {
	Model model;
	Shader shader;

	GLint model_location, view_location, projection_location;

	Container(glm::vec3 size):model("./resources/Container/container.obj",size,glm::vec3(size.x/2.f, 0,size.z/2.f) ),
		shader(Shader::SHADERS_PATH("Container_vs.glsl").c_str(), Shader::SHADERS_PATH("Container_fs.glsl").c_str(),nullptr) {
		
		model_location = glGetUniformLocation(shader.Program, "model");
		view_location = glGetUniformLocation(shader.Program, "view");
		projection_location = glGetUniformLocation(shader.Program, "projection");
	}

	void draw(const DrawCommand& drawCommand) {

		glm::vec3 cameraPos = drawCommand.cameraPosition;

		shader.Use();
		glUniformMatrix4fv(view_location, 1, GL_FALSE, (const GLfloat*)glm::value_ptr(drawCommand.view));
		glUniformMatrix4fv(projection_location, 1, GL_FALSE, (const GLfloat*)glm::value_ptr(drawCommand.projection));

		glUniform3f(glGetUniformLocation(shader.Program, "cameraPosition"), cameraPos.x, cameraPos.y, cameraPos.z);

		model.Draw(shader, true);
	}
};