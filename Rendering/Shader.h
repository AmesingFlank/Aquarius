
#ifndef AQUARIUS_SHADER_H
#define AQUARIUS_SHADER_H

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "../Common/GpuCommons.h"
#include <vector>
class Shader
{
public:
	static std::string SHADERS_PATH(const std::string& file = "") { 
		return "./resources/Shaders/" + file; 
	}

	static std::string readTextFile(const std::string& path) {
		std::string result;
		std::ifstream file;
		file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
		file.open(path);

		std::stringstream stream;

		stream << file.rdbuf();

		result = stream.str();
		return result;
	}

	static std::string readTextFiles(const std::vector<std::string>& paths) {
		std::string result = "";
		for (const std::string& path : paths) {
			result += readTextFile(path) + "\n";
		}

		return result;
	}

	static std::string getVersionString() {
		return "#version 330 core\n";
	}



public:
    GLuint Program;
	Shader(const std::string& vertexPath, const std::string& fragmentPath,
		const std::vector<std::string>& fragmentAdditionalPaths = std::vector<std::string>());

	void Use() { glUseProgram(this->Program); }

private:
    void checkCompileErrors(GLuint shader, std::string type)
    {
        GLint success;
        GLchar infoLog[1024];
        if(type != "PROGRAM")
        {
            glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
            if(!success)
            {
                glGetShaderInfoLog(shader, 1024, NULL, infoLog);
                std::cout << "| ERROR::::SHADER-COMPILATION-ERROR of type: " << type << "|\n" << infoLog << "\n| -- --------------------------------------------------- -- |" << std::endl;
            }
        }
        else
        {
            glGetProgramiv(shader, GL_LINK_STATUS, &success);
            if(!success)
            {
                glGetProgramInfoLog(shader, 1024, NULL, infoLog);
                std::cout << "| ERROR::::PROGRAM-LINKING-ERROR of type: " << type << "|\n" << infoLog << "\n| -- --------------------------------------------------- -- |" << std::endl;
            }
        }
    }
};

#endif //AQUARIUS_SHADER_H
