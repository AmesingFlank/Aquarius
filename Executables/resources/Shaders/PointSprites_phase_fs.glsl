
in vec4 VolumeFractions;

layout(location = 0) out vec4 phaseThickness;

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
	coord2D = gl_PointCoord * 2.0 - vec2(1);

	float distanceToCenter = length(coord2D);
	if (distanceToCenter >= 1.0)
		discard;

	phaseThickness = VolumeFractions;



}