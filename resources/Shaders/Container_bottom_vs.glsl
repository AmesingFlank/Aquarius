layout(location = 0) in vec3 position;


uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 FragPos;


void main()
{
	vec4 FragPos4 = model * vec4(position, 1.0f);
	FragPos = FragPos4.xyz / FragPos4.w;
	gl_Position = projection * view * model * vec4(position, 1.0f);



}
