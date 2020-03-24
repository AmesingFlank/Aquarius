out vec4 color;
in vec3 FragPos; 

uniform vec3 cameraPosition;
uniform vec3 lightPosition;
uniform float containerSize;
uniform float cornellBoxSize;
uniform int environmentMode;
uniform samplerCube skybox;


void main() {
	
	color = vec4(1);
	color = rayTraceEnvironment(cameraPosition, normalize(FragPos-cameraPosition), environmentMode,  cornellBoxSize, containerSize, lightPosition, skybox);


}