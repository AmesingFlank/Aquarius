in vec3 fragPos;
out vec4 color;
in vec3 fragPosNDC;
in vec3 fragNormal;

uniform vec3 cameraPosition;
uniform vec3 lightPosition;
uniform float containerSize;
uniform float cornellBoxSize;
uniform int environmentMode;
uniform samplerCube skybox;
uniform sampler2D oxLogo;
uniform int renderMode;

uniform mat4 inverseView;



uniform sampler2D phaseThicknessTexture;
uniform int usePhaseThicknessTexture;

uniform vec4 phaseColors[4];
uniform int phaseCount;

#define SCREEN_SPACE_NORMAL 0

#if SCREEN_SPACE_NORMAL
uniform sampler2D normalTexture;
#endif


#define RENDER_FLAT 2
#define RENDER_NORMAL 3
#define RENDER_MIRROR 4

#define RENDER_REFRACT 5
#define RENDER_0 6
#define RENDER_1 7


vec4 traceRay(vec3 origin, vec3 direction) {
	return rayTraceEnvironment(origin,direction, environmentMode, cornellBoxSize, containerSize, lightPosition, skybox,oxLogo);
}

float schlick(vec3 normal, vec3 incident) {
	float waterIOR = 1.333;
	float airIOR = 1;
	float F0 = (waterIOR - airIOR) / (waterIOR + airIOR);
	F0 = F0 * F0;
	float F = F0 + (1 - F0) * pow((1 - dot(normal, -incident)), 1);
	return F;
}


vec3 getRefractedRay(vec3 normal, vec3 incident) {
	float waterIOR = 1.333;
	float n = 1 / waterIOR;
	float w = n * dot(normal, -incident);
	float k = sqrt(1 + (w - n) * (w + n));
	vec3 t = (w - k) * normal + n * incident;
	return normalize(t);
}

#if SCREEN_SPACE_NORMAL

vec3 getNormalFromScreenSpaceTexture() {
	vec2 texCoords = (fragPosNDC.xy + vec2(1, 1)) / 2;
	vec4 normal1 = texture(normalTexture, texCoords);
	vec3 normal = normal1.xyz / normal1.w;
	normal = mat3(inverseView) * normal;
	return normal;
}

#endif

vec4 getReflectedColor(vec3 reflectedRay,vec3 fragPos,vec3 normal) {
	vec4 color = traceRay(fragPos, reflectedRay);
	vec3 fragToLight = normalize(lightPosition - fragPos);
	
	if (environmentMode == ENVIRONMENT_CHESS_BOARD || environmentMode == ENVIRONMENT_CORNELL_BOX) {
		float r = 0.3;
		color = color * ((1-r)+r*dot(normal,fragToLight));
	}
	return color;
	
}

vec4 lambertian(vec3 pos, vec3 normal) {
	vec3 fragToLight = normalize(lightPosition - pos);
	return vec4(vec3(0.3 + 0.5 * dot(normal, fragToLight)),1);
}

void main()
{

	vec2 texCoords = (fragPosNDC.xy + vec2(1, 1)) / 2;
	

	vec3 normal = normalize(fragNormal);

#if SCREEN_SPACE_NORMAL
	normal = getNormalFromScreenSpaceTexture();
	color = vec4(normal, 1); 
#endif

	if (renderMode == RENDER_FLAT) {
		color = lambertian(fragPos, normal);
		return;
	}
	if (renderMode == RENDER_NORMAL) {
		color = vec4(normal,1);
		return;
	}

	


	vec3 incident = normalize(fragPos - cameraPosition);

	vec3 reflectedRay = reflect(incident, normal);

	vec4 reflectColor = getReflectedColor(reflectedRay,fragPos,normal);

	if (renderMode == RENDER_MIRROR) {
		color = reflectColor;
		return;
	}

	vec3 refractedRay = getRefractedRay(normal,incident);

	
	vec4 refractColor = vec4(0);

	

	if (usePhaseThicknessTexture > 0) {
		vec4 phaseThickness = texture (phaseThicknessTexture, texCoords) * 0.1;
		float thickness = phaseThickness.x + phaseThickness.y + phaseThickness.z + phaseThickness.w;

		float fractions[4];
		fractions[0] = phaseThickness.x / thickness;
		fractions[1] = phaseThickness.y / thickness;
		fractions[2] = phaseThickness.z / thickness;
		fractions[3] = phaseThickness.w / thickness;

		vec4 tintColor = vec4(0);


		float colorThickness = 
			phaseThickness.x * phaseColors[0].a + 
			phaseThickness.y * phaseColors[1].a +
			phaseThickness.z * phaseColors[2].a +
			phaseThickness.w * phaseColors[3].a;

		if (colorThickness > 0) {
			float colorFractions[4];
			colorFractions[0] = phaseThickness.x * phaseColors[0].a / colorThickness;
			colorFractions[1] = phaseThickness.y * phaseColors[1].a / colorThickness;
			colorFractions[2] = phaseThickness.z * phaseColors[2].a / colorThickness;
			colorFractions[3] = phaseThickness.w * phaseColors[3].a / colorThickness;

			for (int i = 0; i < phaseCount; ++i) {
				tintColor.rgb += colorFractions[i] * phaseColors[i].rgb;
				tintColor.a += fractions[i] * phaseColors[i].a;
			}
		}
		
		float attenuate = max(exp(0.2 * -colorThickness) ,0.2);

		
		refractColor = mix(tintColor, traceRay(fragPos, refractedRay), attenuate);

		if (renderMode == RENDER_REFRACT) {
			color = refractColor;
			return;
		}
		if (renderMode == RENDER_0) {
			color = vec4(vec3(phaseThickness.x/50),1);
			return;
		}
		if (renderMode == RENDER_1) {
			color = vec4(vec3(phaseThickness.y/50), 1);
			return;
		}
	}
	else {
		float attenuate = 0.5;
		vec4 tintColor = vec4(6, 105, 217,256) / 256;
		refractColor = mix(tintColor, traceRay(fragPos, refractedRay), attenuate);
	}


	float mixFactor = schlick(normal, incident);
	//mixFactor = 0.7;
	//mixFactor = min(0.5, max(mixFactor,0.1));

	color =  mix(refractColor, reflectColor, mixFactor);

	color.a = 1;

	//color.rgb = normalize(normal);



	return;


}



