
#version 330 core
in vec3 TexCoords;
layout (location = 0) out vec4 thicknessOutput;

in vec3 debug;

void main()
{  
	vec2 coord2D;
	coord2D = gl_PointCoord* 2.0 - vec2(1); 

	float distanceToCenter = length(coord2D);
	if(distanceToCenter >= 1.0) 
     discard;

	float zInSphere = sqrt(1-coord2D.x*coord2D.x - coord2D.y * coord2D.y);

	thicknessOutput = vec4(zInSphere * 2,0,0,1);

}

