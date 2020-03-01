in vec4 Color;

out vec4 FragColor;

in vec3 debug;

uniform float radius;

in float sphereRadius;
in vec4 posToCamera;
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


	vec3 coordInSphere = vec3(coord2D,zInSphere);
	vec3 lightDir = vec3(0,1,0);

	float diffuse = max(0.0, dot(lightDir, coordInSphere));
	diffuse = 0.5 + diffuse / 2 ;
	FragColor = vec4(diffuse * vec3(0,0,1),1);

	float depth = (posToCamera.z / posToCamera.w) + zInSphere * sphereRadius;


	gl_FragDepth = projectZ(depth);
	gl_FragDepth = 0.5 * (gl_DepthRange.diff * gl_FragDepth + gl_DepthRange.far + gl_DepthRange.near);

}
