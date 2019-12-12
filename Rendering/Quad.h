//
// Created by AmesingFlank on 2019-04-22.
//

#ifndef AQUARIUS_QUAD_H
#define AQUARIUS_QUAD_H

#include "Shader.h"


struct Quad {

    float coords[30]={-1.f,1.f,0.f,  0.f,1.f,
                      -1.f,-1.f,0.f,  0.f,0.f,
                      1.f,-1.f,0.f,  1.f,0.f,

                      -1.f,1.f,0.f,  0.f,1.f,
                      1.f,-1.f,0.f,  1.f,0.f,
                      1.f,1.f,0.f,  1.f,1.f,
    };


    Shader* shader;

    GLuint VBO,VAO, vertex_shader, fragment_shader;
    GLint vPos_location, texCoord_location,quadTexture_location;

    Quad (){

		std::string vsPath = Shader::SHADERS_PATH("Quad_vs.glsl");
		std::string fsPath = Shader::SHADERS_PATH("Quad_fs.glsl");

        shader = new Shader(vsPath.c_str(),fsPath.c_str(), nullptr);

        vPos_location = glGetAttribLocation(shader->Program, "vPos");
        texCoord_location = glGetAttribLocation(shader->Program,"texCoord");
        quadTexture_location = glGetUniformLocation(shader->Program,"quadTexture");

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

        glUseProgram(shader->Program);
        glBindVertexArray(VAO);
        glActiveTexture(GL_TEXTURE0);
        glUniform1i(quadTexture_location,0);
        glBindTexture(GL_TEXTURE_2D,texture);

        glDrawArrays(GL_TRIANGLES,0,6);

        glBindVertexArray(0);

    }
};

#endif //AQUARIUS_QUAD_H
