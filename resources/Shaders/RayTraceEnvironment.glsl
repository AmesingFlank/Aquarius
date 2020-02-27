struct Ray {
	vec3 origin;
	vec3 invDir;
	vec3 dir;
};

struct Hit {
	float t;
	vec3 hitPos;
	bool hit;
};


void boxIntersect(vec3 boxMin, vec3 boxMax, Ray r, out Hit hit) {
	vec3 tbot = r.invDir * (boxMin - r.origin);
	vec3 ttop = r.invDir * (boxMax - r.origin);
	vec3 tmin = min(ttop, tbot);
	vec3 tmax = max(ttop, tbot);
	vec2 t = max(tmin.xx, tmin.yz);
	float t0 = max(t.x, t.y);
	t = min(tmax.xx, tmax.yz);
	float t1 = min(t.x, t.y);

	if (t1 < 0) {
		hit.hit = false;
	}

	hit.t = t0;
	if (t0 < 0) {
		hit.t = t1;
	}
	hit.hitPos = r.origin + hit.t * r.dir;

	hit.hit = true;
}


vec3 getCornellColor(vec3 pos,float xmin,float xmax) {
	if (abs(pos.x - xmin) < 1e-3) {
		return vec3(1, 0, 0);
	}
	else if (abs(pos.x - xmax) < 1e-3) {
		return vec3(0, 0, 1);
	} 
	return vec3(0.8);
}

vec3 getCornellNormal(vec3 pos, vec3 boxMin, vec3 boxMax) {
	if (abs(pos.x - boxMin.x) < 1e-3) {
		return vec3(1, 0, 0);
	}
	else if (abs(pos.x - boxMax.x) < 1e-3) {
		return vec3(-1, 0, 0);
	}
	else if (abs(pos.y - boxMin.y) < 1e-3) {
		return vec3(0, 1, 0);
	}
	else if (abs(pos.y - boxMax.y) < 1e-3) {
		return vec3(0, -1, 0);
	}
	else if (abs(pos.z - boxMin.z) < 1e-3) {
		return vec3(0, 0, 1);
	}
	else if (abs(pos.z - boxMax.z) < 1e-3) {
		return vec3(0, 0, -1);
	}
}


vec3 shadeCornell(vec3 pos, vec3 boxMin, vec3 boxMax,vec3 lightPos) {
	vec3 baseColor = getCornellColor(pos, boxMin.x, boxMax.x);
	vec3 normal = getCornellNormal(pos, boxMin, boxMax);
	vec3 fragToLight = normalize(lightPos - pos);

	vec3 result = baseColor * (0.5 + 0.5 * dot(normal, fragToLight));
	return result;
}


vec3 rayTraceEnvironment(vec3 cameraPos, vec3 direction, vec3 lightPos,float boxSize) {
	Hit hit;

	vec3 boxMin = vec3(-boxSize, 0, -boxSize);
	vec3 boxMax = vec3(2*boxSize, 3*boxSize, 3*boxSize);

	Ray ray;
	ray.origin = cameraPos;
	ray.dir = direction;
	ray.invDir.x = 1.0 / direction.x;
	ray.invDir.y = 1.0 / direction.y;
	ray.invDir.z = 1.0 / direction.z;

	boxIntersect(boxMin, boxMax, ray, hit);


	return shadeCornell(hit.hitPos, boxMin,boxMax,lightPos);
}


float spotLightIntensity(vec3 fragToLight, vec3 normal) {
	float cosTheta = dot(fragToLight, normal);

	float threshold1 = 0.95;
	float threshold2 = 0.945;

	float intensity1 = 1;
	float intensity2 = 0.1;

	if (cosTheta > threshold1) {
		return intensity1;
	}
	if (cosTheta > threshold2) {
		float t = (cosTheta - threshold2) / (threshold1 - threshold2);
		return t * intensity1 + (1 - t) * intensity2;
	}
	return intensity2;
}


vec4 rayTraceEnvironmentSpotLight(vec3 cameraPos, vec3 direction, vec3 lightPos) {
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