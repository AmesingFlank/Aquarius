layout (location = 0) in vec3 position;
layout (location = 1) in vec4 volumeFractions;
out vec4 VolumeFractions;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

uniform float windowWidth;
uniform float windowHeight;
uniform float radius;

uniform float tanHalfFOV;

uniform vec3 cameraPosition;

out vec3 debug;
out vec4 posToCamera;
out float sphereRadius;
out mat4 proj;



void main()
{

	mat4 mvp = projection * view * model;
	
	vec3 cameraToPoint = position-cameraPosition;
	float distance = length(cameraToPoint);

	float sizeAtDistanceForHalfScreen = tanHalfFOV * distance;
	gl_PointSize = (windowHeight / 2) * radius / sizeAtDistanceForHalfScreen;
	

    gl_Position =   mvp * vec4(position, 1.0);
	proj = projection;
	sphereRadius = radius;
	posToCamera = view * model * vec4(position, 1.0);

	VolumeFractions = volumeFractions;
}
