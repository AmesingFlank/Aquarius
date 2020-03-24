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

GLint Shader::getUniformLocation(std::string name) {
	return glGetUniformLocation(this->program, name.c_str());
}


void checkUniformLocationError(int loc,std::string name) {
	if (loc == -1) {
		std::cout <<"ERROR: Uniform Not Found: "<< name << std::endl;
	}
}

void Shader::setUniform1i(std::string name, int val,bool debug) {
	int loc = getUniformLocation(name);
	if(debug) checkUniformLocationError(loc, name);
	glUniform1i(loc, val);
}
void Shader::setUniform1f(std::string name, float val, bool debug) {
	int loc = getUniformLocation(name);
	if (debug) checkUniformLocationError(loc, name);
	glUniform1f(loc, val);
}

void Shader::setUniformMat4(std::string name, const glm::mat4& mat, bool debug) {
	int loc = getUniformLocation(name);
	if (debug) checkUniformLocationError(loc, name);
	glUniformMatrix4fv(loc, 1, GL_FALSE, (const GLfloat*)glm::value_ptr(mat));
}
void Shader::setUniform3f(std::string name, float3 val, bool debug) {
	int loc = getUniformLocation(name);
	if (debug) checkUniformLocationError(loc, name);
	glUniform3f(loc, val.x,val.y,val.z);
}
void Shader::setUniform3f(std::string name, glm::vec3 val, bool debug) {
	int loc = getUniformLocation(name);
	if (debug) checkUniformLocationError(loc, name);
	glUniform3f(loc, val.x, val.y, val.z);
}


void Shader::setUniform4f(std::string name, float4 val, bool debug) {
	int loc = getUniformLocation(name);
	if (debug) checkUniformLocationError(loc, name);
	glUniform4f(loc, val.x, val.y, val.z,val.w);
}
void Shader::setUniform4f(std::string name, glm::vec4 val, bool debug) {
	int loc = getUniformLocation(name);
	if (debug) checkUniformLocationError(loc, name);
	glUniform4f(loc, val.x, val.y, val.z,val.w);
}

void Shader::setUniformDrawCommand(const DrawCommand& drawCommand, bool debug) {
	setUniformMat4("view", drawCommand.view);
	setUniformMat4("projection", drawCommand.projection);

	

	setUniform3f("cameraPosition", drawCommand.cameraPosition);

	setUniform1f("windowWidth", drawCommand.windowWidth);

	setUniform1f("windowHeight", drawCommand.windowHeight);

	float tanHalfFOV = tan(glm::radians(drawCommand.FOV) / 2);
	setUniform1f("tanHalfFOV", tanHalfFOV);

	setUniform3f("lightPosition", drawCommand.lightPosition);

	setUniform1f("containerSize", drawCommand.containerSize);
	setUniform1f("cornellBoxSize", drawCommand.cornellBoxSize);
	setUniform1i("environmentMode", (int)drawCommand.environmentMode);

	setUniform1i("renderMode", (int)drawCommand.renderMode);
}