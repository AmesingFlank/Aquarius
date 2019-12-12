
#version 330 core
in vec3 TexCoords;
layout (location = 0) out vec4 depthOutput;

in vec4 posToCamera;
in float sphereRadius;

in mat4 proj;

float projectZ(float viewZ) {
	vec3 dummyViewSpacePoint = vec3(0, 0, viewZ);
	vec4 projected = proj * vec4(dummyViewSpacePoint, 1);
	return projected.z / projected.w;
}

void main()
{  
	vec2 coord2D;
	coord2D = gl_PointCoord* 2.0 - vec2(1); 

	float distanceToCenter = length(coord2D);
	if(distanceToCenter >= 1.0) 
     discard;

	float zInSphere = sqrt(1-coord2D.x*coord2D.x - coord2D.y * coord2D.y);

	float depth = (posToCamera.z / posToCamera.w) + zInSphere * sphereRadius;

	depthOutput.r = depth;

	gl_FragDepth = projectZ(depth);
	gl_FragDepth = 0.5 * (gl_DepthRange.diff * gl_FragDepth + gl_DepthRange.far + gl_DepthRange.near);

}

