#version 330 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec4 volumeFractions;
out vec3 TexCoords;
out vec4 VolumeFractions;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

uniform float windowWidth;
uniform float windowHeight;
uniform float radius;

uniform vec3 cameraPosition;

out vec3 debug;
out vec4 posToCamera;
out float sphereRadius;
out mat4 proj;



void main()
{

	mat4 mvp = projection * view * model;
	
	vec3 cameraToPoint = position-cameraPosition;

	vec3 up = vec3(0,1,0);
	vec3 right = cross(up,cameraToPoint );
	vec3 planeUp = cross(right,cameraToPoint);
	planeUp = normalize(planeUp);
	
	vec3 topPosition = position + planeUp * radius;
	vec3 bottomPosition = position - planeUp * radius;

	vec4 topProjected = mvp * vec4(topPosition,1);
	vec4 bottomProjected = mvp * vec4(bottomPosition,1);

	topProjected = topProjected / topProjected.w;
	bottomProjected = bottomProjected / bottomProjected.w;

	float diff = abs(topProjected.y-bottomProjected.y);

	float diffInPixels =   diff*windowHeight/2; //divide by 2 because NDC goes from -1 to 1

	gl_PointSize = diffInPixels;

    gl_Position =   mvp * vec4(position, 1.0);
    TexCoords = position;
	proj = projection;
	sphereRadius = radius;
	posToCamera = view * model * vec4(position, 1.0);

	VolumeFractions = volumeFractions;
}
