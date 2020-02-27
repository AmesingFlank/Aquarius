out vec4 color;
in vec3 FragPos; 

uniform vec3 cameraPos;
uniform vec3 lightPos;
uniform float boxSize;
void main() {
	
	color = vec4(1);
	color.rgb = rayTraceEnvironment(cameraPos, normalize(FragPos-cameraPos), lightPos,boxSize);

}