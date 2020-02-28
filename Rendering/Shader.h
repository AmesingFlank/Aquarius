
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
    GLuint program;
	Shader(const std::string& vertexPath, const std::string& fragmentPath,
		const std::vector<std::string>& fragmentAdditionalPaths = std::vector<std::string>());

	void use() { glUseProgram(this->program); }

	~Shader();

private:
	void checkCompileErrors(GLuint shader, std::string type);



};

#endif //AQUARIUS_SHADER_H
