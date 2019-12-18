#version 330 core
layout(location = 0) in vec3 position;

out vec3 fragPos;
out vec3 fragPosNDC;
out vec3 fragPosViewSpace;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
	fragPos = (model * vec4(position, 1)).xyz;

	gl_Position = projection * view * model * vec4(position, 1.0f);

	fragPosNDC = gl_Position.xyz / gl_Position.w;

	fragPosViewSpace = (view * model * vec4(position, 1.0f)).xyz;
}
