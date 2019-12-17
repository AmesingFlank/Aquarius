#version 330 core
in vec3 FragPos;
out vec4 color;

void main()
{
	float intensity = 0.2 + 0.8*(FragPos.y)/5;
	color = vec4(FragPos.x/10,FragPos.y/5,FragPos.z/10, 1);

	color *= intensity;
	color.a = 1;

}
