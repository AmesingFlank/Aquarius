out vec4 color;
in vec3 FragPos; 

uniform vec3 cameraPos;
uniform vec3 lightPos;

void main() {
	
	color = vec4(1);
	color = rayTraceEnvironment(cameraPos, normalize(FragPos-cameraPos), lightPos);

}