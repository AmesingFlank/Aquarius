#pragma once

#include "../../GpuCommons.h"
#include "PointSprites_vs.glsl"
#include "PointSprites_fs.glsl"
#include "../Shader.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>


struct PointSprites {
	int count;
	float* positionsHost;
	GLuint VAO, VBO;
	GLint model_location, view_location, projection_location, vPos_location;

	cudaGraphicsResource* cudaResourceVBO;
	float* positionsDevice;

	Shader* shader;

	PointSprites(int count_):count(count_) {

		positionsHost = new float[count*3];

		std::string vs = PointSprites_vs;
		std::string fs = PointSprites_fs;

		shader = new Shader(1, vs.c_str(), fs.c_str(), nullptr);


		vPos_location = glGetAttribLocation(shader->Program, "position");
		model_location = glGetUniformLocation(shader->Program, "model");
		view_location = glGetUniformLocation(shader->Program, "view");
		projection_location = glGetUniformLocation(shader->Program, "projection");


		glGenVertexArrays(1, &VAO);
		glGenBuffers(1, &VBO);
		glBindVertexArray(VAO); 
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float)*count*3, positionsHost, GL_STATIC_DRAW);
		glEnableVertexAttribArray(vPos_location);
		glVertexAttribPointer(vPos_location, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, 0);

		HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&cudaResourceVBO, VBO, cudaGraphicsMapFlagsNone));
		
		size_t  size;   
		HANDLE_ERROR(cudaGraphicsMapResources(1, &cudaResourceVBO, NULL));
		HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&positionsDevice, &size, cudaResourceVBO));

		glBindVertexArray(0);



	}

	glm::mat4  model = glm::mat4(1.0);

	void draw(glm::mat4& view, glm::mat4& projection,glm::vec3 cameraPos,float windowWidth,float windowHeight,float radius) {

		shader->Use();
		glUniformMatrix4fv(model_location, 1, GL_FALSE, (const GLfloat*)glm::value_ptr(model));
		glUniformMatrix4fv(view_location, 1, GL_FALSE, (const GLfloat*)glm::value_ptr(view));
		glUniformMatrix4fv(projection_location, 1, GL_FALSE, (const GLfloat*)glm::value_ptr(projection));

		glUniform1f(glGetUniformLocation(shader->Program, "windowWidth"), windowWidth);
		glUniform1f(glGetUniformLocation(shader->Program, "windowHeight"), windowHeight);

		glUniform1f(glGetUniformLocation(shader->Program, "radius"), radius);

		glUniform3f(glGetUniformLocation(shader->Program, "cameraPosition"), cameraPos.x, cameraPos.y, cameraPos.z);


		glBindVertexArray(VAO);
		glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
		glPointParameteri(GL_POINT_SPRITE_COORD_ORIGIN, GL_LOWER_LEFT);
		//glPointSize(50);
		glDrawArrays(GL_POINTS, 0, count);
	}
};