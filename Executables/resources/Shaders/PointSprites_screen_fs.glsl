

// Shader for Screen Space Fluids.
// Deprecated. The project is using a mesh based renderer now.

in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D depthTextureNDC;
uniform sampler2D depthTexture;
uniform sampler2D normalTexture;
uniform sampler2D thicknessTexture;

uniform mat4 inverseView;
uniform mat4 projection;
uniform float windowWidth;
uniform float windowHeight;
uniform float FOV;




uniform vec3 cameraPosition;
uniform vec3 lightPosition;
uniform float containerSize;
uniform float cornellBoxSize;
uniform int environmentMode;
uniform samplerCube skybox;
uniform int renderMode;

#define M_PI 3.1415926535897932384626433832795

float projectZ(float viewZ) {
	vec3 dummyViewSpacePoint = vec3(0, 0, viewZ);
	vec4 projected = projection * vec4(dummyViewSpacePoint,1);
	return projected.z / projected.w;
}

vec3 getViewPos(float depth) {
	float xPhysicalHalfScreen = tan(FOV * M_PI / 180) * abs(depth);
	float yPhysicalHalfScreen = xPhysicalHalfScreen * windowHeight / windowWidth;

	float x = (TexCoord.x * 2 - 1) * xPhysicalHalfScreen;
	float y = (TexCoord.y * 2 - 1) * yPhysicalHalfScreen;

	return vec3(x, y, depth);
}

vec3 getWorldPos(float depth) {
	vec3 viewPos = getViewPos(depth);
	return (inverseView * vec4(viewPos,1)).xyz;
}

vec3 getViewNormal() {
	return normalize(texture (normalTexture, TexCoord).rgb);
}

vec3 getWorldNormal() {
	vec3 viewNormal = getViewNormal();
	vec3 worldNormal = mat3(inverseView) * viewNormal;
	return normalize(worldNormal);
}

vec4 traceRay(vec3 origin, vec3 direction) {
	return rayTraceEnvironment(origin, direction, environmentMode, cornellBoxSize, containerSize, lightPosition, skybox);
}


void main() {
	
	float depth = texture (depthTexture,TexCoord).r;

	
	if(depth>=0){
		discard;return;
	}

	gl_FragDepth = projectZ(depth);
	gl_FragDepth = 0.5 * (gl_DepthRange.diff * gl_FragDepth + gl_DepthRange.far + gl_DepthRange.near);



	vec4 thicknessAllPhases = texture (thicknessTexture, TexCoord);
	float thickness = thicknessAllPhases.r + thicknessAllPhases.g + thicknessAllPhases.b + thicknessAllPhases.a;
	float thicknessHat = 50;
	//thickness = min(thicknessHat, thickness) / thicknessHat;
	//thickness = 0.5 + thickness / 2;

	thickness /= 500;

	vec3 normal = getWorldNormal();

	vec3 fragPos = getWorldPos(depth);

	vec3 incident = normalize(fragPos - cameraPosition);

	vec3 reflectedRay = reflect(incident, normal);

	vec3 reflectColor = traceRay(fragPos,reflectedRay).rgb;

	vec3 refractedRay = normalize(incident - 0.2 * normal);

	float attenuate = max(exp(0.5 * -thickness), 0.2);

	vec3 tint_color = vec3(6, 105, 217) / 256;

	vec3 refractColor = mix(tint_color, traceRay(fragPos, refractedRay).rgb, attenuate);

	float mixFactor = 0.5;

	FragColor = vec4(mix( refractColor, reflectColor, mixFactor),1);
    
}
