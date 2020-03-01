
#ifndef AQUARIUS_QUAD_H
#define AQUARIUS_QUAD_H

#include "Shader.h"
#include <memory>

struct Quad {

    float coords[30]={-1.f,1.f,0.f,  0.f,1.f,
                      -1.f,-1.f,0.f,  0.f,0.f,
                      1.f,-1.f,0.f,  1.f,0.f,

                      -1.f,1.f,0.f,  0.f,1.f,
                      1.f,-1.f,0.f,  1.f,0.f,
                      1.f,1.f,0.f,  1.f,1.f,
    };


    std::shared_ptr<Shader> shader;

    GLuint VBO,VAO;
    GLint vPos_location, texCoord_location;

    Quad (){

		std::string vsPath = Shader::SHADERS_PATH("Quad_vs.glsl");
		std::string fsPath = Shader::SHADERS_PATH("Quad_fs.glsl");

        shader = std::make_shared<Shader>(vsPath,fsPath);

        vPos_location = glGetAttribLocation(shader->program, "vPos");
        texCoord_location = glGetAttribLocation(shader->program,"texCoord");

        glGenVertexArrays(1,&VAO);
        glGenBuffers(1, &VBO);
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(coords), coords, GL_STATIC_DRAW);
        glEnableVertexAttribArray(vPos_location);
        glVertexAttribPointer(vPos_location,3,GL_FLOAT,GL_FALSE,sizeof(float) * 5, (void* )(sizeof(float) * 0));
        glEnableVertexAttribArray(texCoord_location);
        glVertexAttribPointer(texCoord_location,2,GL_FLOAT,GL_FALSE,sizeof(float) * 5, (void* )(sizeof(float) * 3));
        glBindVertexArray(0);

    }
    void draw(GLuint texture){

        glUseProgram(shader->program);
        glBindVertexArray(VAO);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D,texture);
		shader->setUniform1i("quadTexture", 0);

        glDrawArrays(GL_TRIANGLES,0,6);

        glBindVertexArray(0);

    }

	~Quad() {
		glDeleteBuffers(1, &VBO);
		glDeleteVertexArrays(1, &VAO);
	}
	
};

#endif //AQUARIUS_QUAD_H
