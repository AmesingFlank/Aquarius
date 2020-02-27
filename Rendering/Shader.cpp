#include "Shader.h"

Shader::Shader(const std::string& vertexPath, const std::string& fragmentPath, 
	const std::vector<std::string>& fragmentAdditionalPaths)
{
	std::string vertexCode;
	std::string fragmentCode;


	try
	{
		vertexCode = getVersionString()+readTextFile(vertexPath);
		fragmentCode = getVersionString()+ readTextFiles(fragmentAdditionalPaths)+readTextFile(fragmentPath);
	}
	catch (std::ifstream::failure e)
	{
		std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ" << std::endl;
	}
	const GLchar* vShaderCode = vertexCode.c_str();
	const GLchar* fShaderCode = fragmentCode.c_str();

	GLuint vertex, fragment;
	GLint success;
	GLchar infoLog[512];

	vertex = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex, 1, &vShaderCode, NULL);
	glCompileShader(vertex);
	checkCompileErrors(vertex, "VERTEX");

	fragment = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment, 1, &fShaderCode, NULL);
	glCompileShader(fragment);
	checkCompileErrors(fragment, "FRAGMENT");



	this->Program = glCreateProgram();
	glAttachShader(this->Program, vertex);
	glAttachShader(this->Program, fragment);


	glLinkProgram(this->Program);
	checkCompileErrors(this->Program, "PROGRAM");

	glDeleteShader(vertex);
	glDeleteShader(fragment);



}