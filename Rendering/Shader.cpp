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



	this->program = glCreateProgram();
	glAttachShader(this->program, vertex);
	glAttachShader(this->program, fragment);


	glLinkProgram(this->program);
	checkCompileErrors(this->program, "PROGRAM");

	glDeleteShader(vertex);
	glDeleteShader(fragment);


}

void Shader::checkCompileErrors(GLuint shader, std::string type)
{
	GLint success;
	GLchar infoLog[1024];
	if (type != "PROGRAM")
	{
		glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			glGetShaderInfoLog(shader, 1024, NULL, infoLog);
			std::cout << "| ERROR::::SHADER-COMPILATION-ERROR of type: " << type << "|\n" << infoLog << "\n| -- --------------------------------------------------- -- |" << std::endl;
		}
	}
	else
	{
		glGetProgramiv(shader, GL_LINK_STATUS, &success);
		if (!success)
		{
			glGetProgramInfoLog(shader, 1024, NULL, infoLog);
			std::cout << "| ERROR::::PROGRAM-LINKING-ERROR of type: " << type << "|\n" << infoLog << "\n| -- --------------------------------------------------- -- |" << std::endl;
		}
	}
}

Shader::~Shader() {
	glDeleteProgram(this->program);
}