#version 330 core
in vec3 fragPos;
out vec4 color;
in vec3 fragPosNDC;
in vec3 fragNormal;

uniform sampler2D normalTexture;

uniform vec3 cameraPosition;
uniform samplerCube skybox;
uniform mat4 inverseView;


vec3 traceRay(vec3 origin, vec3 direction) {
	float tHitGround = origin.y / -direction.y;
	if (tHitGround > 0) {
		vec3 hitPos = origin + tHitGround * direction;
		if (hitPos.x >= 0 && hitPos.x <= 10 && hitPos.z >= 0 && hitPos.z <= 10) {
			int xi = int(hitPos.x);
			int zi = int(hitPos.z);
			if ((xi + zi) % 2 == 0)
				return vec3(0.5, 0.5, 0.5);
			else
				return vec3(0.8, 0.8, 0.8);
		}
	}
	return texture(skybox, direction).rgb;
}

void main()
{


	vec2 texCoords = (fragPosNDC.xy + vec2(1, 1)) / 2;

	vec3 viewSpaceNormal = texture(normalTexture, texCoords).rgb;
	vec3 normal = mat3(inverseView) * viewSpaceNormal;

	normal = fragNormal;

	color.rgb = normal;

	vec3 incident = normalize(fragPos - cameraPosition);

	vec3 reflectedRay = reflect(incident, normal);

	vec3 reflectColor = traceRay(fragPos, reflectedRay);

	vec3 refractedRay = normalize(incident - 0.2 * normal);

	float thickness = 0.5;

	float attenuate = max(exp(0.5 * -thickness), 0.2);

	vec3 tint_color = vec3(6, 105, 217) / 256;

	vec3 refractColor = mix(tint_color, traceRay(fragPos, refractedRay), attenuate);

	float mixFactor = 0.5;

	color = vec4(mix(refractColor, reflectColor, mixFactor), 1);

	//return;


	vec3 lightDir = vec3(0, 1, 0);
	float diffuse = max(dot(normal, lightDir), 0.0);

	float spec = pow(max(dot(-incident, reflectedRay), 0.0), 50);

	vec3 ambient = vec3(0.2);

	color = vec4(ambient + diffuse * vec3(0.4) + spec*vec3(0.4), 1);


}
