out vec4 color;
in vec3 FragPos; 

uniform vec3 cameraPos;
uniform vec3 lightPos;
uniform float boxSize;



vec4 checkerBoard() {
	int xi = int(FragPos.x + 100);
	int zi = int(FragPos.z + 100);
	if ((xi + zi) % 2 == 0)
		return vec4(0.5, 0.5, 0.5, 1);
	else
		return vec4(0.8, 0.8, 0.8, 1);
}



void main() {
	
	color = vec4(1);
	color.rgb = rayTraceEnvironment(cameraPos, normalize(FragPos-cameraPos), lightPos,boxSize);

	color = checkerBoard();

}