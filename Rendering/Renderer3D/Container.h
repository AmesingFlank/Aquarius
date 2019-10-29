#pragma once
#include "../model.h"
#include "Container_vs.glsl"
#include "Container_fs.glsl"

struct Container {
	Model model;
	Shader shader;

	GLint model_location, view_location, projection_location;

	Container(glm::vec3 size):model("./resources/Container/container.obj",size,glm::vec3(size.x/2.f, 0,size.z/2.f) ),
		shader(1,Container_vs.c_str(),Container_fs.c_str(),nullptr) {
		
		model_location = glGetUniformLocation(shader.Program, "model");
		view_location = glGetUniformLocation(shader.Program, "view");
		projection_location = glGetUniformLocation(shader.Program, "projection");
	}

	void draw(glm::mat4& view, glm::mat4& projection, glm::vec3 cameraPos) {

		shader.Use();
		glUniformMatrix4fv(view_location, 1, GL_FALSE, (const GLfloat*)glm::value_ptr(view));
		glUniformMatrix4fv(projection_location, 1, GL_FALSE, (const GLfloat*)glm::value_ptr(projection));

		glUniform3f(glGetUniformLocation(shader.Program, "cameraPosition"), cameraPos.x, cameraPos.y, cameraPos.z);

		model.Draw(shader, true);
	}
};