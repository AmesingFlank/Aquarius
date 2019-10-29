#pragma once
#include<string>
static string Container_vs = R"(
#version 330 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texCoords;

out vec2 TexCoords;
out vec3 Normal;
out vec3 FragPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    FragPos=(model*vec4(position,1)).xyz;
    gl_Position = projection * view * model * vec4(position, 1.0f);
    TexCoords = texCoords;
    Normal=mat3(transpose(inverse(model))) *normal ;
}
)";
