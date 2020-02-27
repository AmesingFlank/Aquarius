layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

out vec3 fragPos;
out vec3 fragPosNDC;
out vec3 fragPosViewSpace;

out vec3 fragNormal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
	fragPos = (model * vec4(position, 1)).xyz;

	gl_Position = projection * view * model * vec4(position, 1.0f);

	fragPosNDC = gl_Position.xyz / gl_Position.w;

	fragPosViewSpace = (view * model * vec4(position, 1.0f)).xyz;

	fragNormal = normal;
}
