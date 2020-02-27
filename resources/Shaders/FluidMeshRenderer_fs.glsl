in vec3 fragPos;
out vec4 color;
in vec3 fragPosNDC;
in vec3 fragNormal;


uniform vec3 cameraPosition;
uniform samplerCube skybox;
uniform mat4 inverseView;



uniform sampler2D phaseThicknessTexture;
uniform int usePhaseThicknessTexture;

uniform vec4 phaseColors[4];
uniform int phaseCount;

vec4 traceRay(vec3 origin, vec3 direction) {
	return vec4(texture(skybox, direction).rgb, 1);
	float tHitGround = origin.y / -direction.y;
	if (tHitGround > 0) {
		vec3 hitPos = origin + tHitGround * direction;
		if (hitPos.x >= -5 && hitPos.x <= 15 && hitPos.z >= -5 && hitPos.z <= 15) {
			int xi = int(hitPos.x + 100);
			int zi = int(hitPos.z + 100);
			if ((xi + zi) % 2 == 0)
				return vec4(0.5, 0.5, 0.5, 1);
			else
				return vec4(0.8, 0.8, 0.8, 1);
		}
	}
	return vec4(texture(skybox, direction).rgb, 1);
}


void main()
{


	vec2 texCoords = (fragPosNDC.xy + vec2(1, 1)) / 2;

	vec3 normal = fragNormal;

	color.rgb = normal;

	vec3 incident = normalize(fragPos - cameraPosition);

	vec3 reflectedRay = reflect(incident, normal);

	vec4 reflectColor = traceRay(fragPos, reflectedRay);

	vec3 refractedRay = normalize(incident - 0.2 * normal);

	
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
		
		float attenuate = max(exp(0.5 * -colorThickness) ,0.2);

		
		refractColor = mix(tintColor, traceRay(fragPos, refractedRay), attenuate);
	}
	else {
		float thickness = 0.5;
		float attenuate = (exp(0.5 * -thickness), 0.2);
		vec4 tintColor = vec4(6, 105, 217,256) / 256;
		refractColor = mix(tintColor, traceRay(fragPos, refractedRay), attenuate);
	}


	float mixFactor = 0.5;

	color =  mix(refractColor, reflectColor, mixFactor);

	color.a = 1;


	return;


}



