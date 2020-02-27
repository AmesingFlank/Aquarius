


float spotLightIntensity(vec3 fragToLight, vec3 normal) {
	float cosTheta = dot(fragToLight, normal);

	float threshold1 = 0.85;
	float threshold2 = 0.82;

	float intensity1 = 1;
	float intensity2 = 0;

	if (cosTheta > threshold1) {
		return intensity1;
	}
	if (cosTheta > threshold2) {
		float t = (cosTheta - threshold2) / (threshold1 - threshold2);
		return t * intensity1 + (1 - t) * intensity2;
	}
	return intensity2;
}


vec4 rayTraceEnvironment(vec3 cameraPos, vec3 direction, vec3 lightPos) {
	float tHitGround = cameraPos.y / -direction.y;
	vec3 hitPos = cameraPos + tHitGround * direction;

	vec3 fragToLight = normalize(lightPos - hitPos);
	vec3 fragToCamera = -direction;

	vec3 normal = vec3(0, 1, 0);

	vec3 ambient = vec3(0.2);
	vec3 diffuse = vec3(0.5) * spotLightIntensity(fragToLight, normal);

	vec3 color = ambient + diffuse;

	return vec4(color, 1);
}