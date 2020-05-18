#define ENVIRONMENT_SKYBOX 0
#define ENVIRONMENT_CORNELL_BOX 1
#define ENVIRONMENT_CHESS_BOARD 2


vec3 chessBoard(vec2 xz) {
	int xi = int(xz.x + 100);
	int zi = int(xz.y + 100);
	if ((xi + zi) % 2 == 0)
		return vec3(0.5, 0.5, 0.5);
	else
		return vec3(0.8, 0.8, 0.8);
}

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

#define EPSILON  1e-3
 
vec3 getCornellColor(vec3 pos, vec3 boxMin, vec3 boxMax,sampler2D oxLogo) {

	if (abs(pos.z - boxMax.z) < EPSILON) {
		float logoSizeRelative = 0.3;
		vec2 logoCenterRelative = vec2(0.5, 0.3);
		vec2 cornellFaceSize = boxMax.xy - boxMin.xy;
		vec2 logoSize = logoSizeRelative * cornellFaceSize;
		vec2 logoCenter = logoCenterRelative * cornellFaceSize + boxMin.xy;

		vec2 logoMin = logoCenter - logoSize / 2;
		vec2 logoMax = logoCenter + logoSize / 2;

		vec2 logoCoord = pos.xy - logoMin;
		logoCoord.x /= logoSize.x;
		logoCoord.y /= logoSize.y;
		logoCoord = vec2(1, 1) - logoCoord;
		if (logoCoord.x >= 0 && logoCoord.x <= 1 && logoCoord.y >= 0 && logoCoord.y <= 1) {
			return texture(oxLogo, logoCoord).rgb;
		}
	}

	if (abs(pos.y - boxMin.y) < EPSILON) {
		return chessBoard(pos.xz);
	}
	return vec3(0.8);
	if (abs(pos.x - boxMin.x) < EPSILON) {
		return vec3(0, 1, 0);
	}
	if (abs(pos.x - boxMax.x) < EPSILON) {
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


vec3 shadeCornell(vec3 pos, vec3 boxMin, vec3 boxMax,vec3 lightPos,sampler2D oxLogo) {
	vec3 baseColor = getCornellColor(pos, boxMin, boxMax,oxLogo);
	vec3 normal = getCornellNormal(pos, boxMin, boxMax);
	vec3 fragToLight = normalize(lightPos - pos);

	vec3 result = baseColor * (0.1 + 0.5 * dot(normal, fragToLight));
	return result;
}


vec4 rayTraceEnvironment(vec3 cameraPos, vec3 direction, int environmentMode,float cornellBoxSize, float containerSize,vec3 lightPos,samplerCube skybox,sampler2D oxLogo) {
	if (environmentMode == ENVIRONMENT_CORNELL_BOX) {
		Hit hit;

		float cornellBoxPadding = (cornellBoxSize - containerSize) / 2;

		vec3 boxMin = vec3(-1, 0, -1) * cornellBoxPadding;
		vec3 boxMax = vec3(2,3,2) * cornellBoxPadding;

		Ray ray;
		ray.origin = cameraPos;
		ray.dir = direction;
		ray.invDir.x = 1.0 / direction.x;
		ray.invDir.y = 1.0 / direction.y;
		ray.invDir.z = 1.0 / direction.z;

		boxIntersect(boxMin, boxMax, ray, hit);


		return vec4(shadeCornell(hit.hitPos, boxMin, boxMax, lightPos,oxLogo),1);
	}
	else {
		float tHitGround = cameraPos.y / -direction.y;
		vec3 hitPos = cameraPos + tHitGround * direction;

		if (environmentMode == ENVIRONMENT_SKYBOX) {
			
			if (tHitGround > 0 &&
				hitPos.x >= 0 &&
				hitPos.x <= containerSize &&
				hitPos.z >= 0 &&
				hitPos.z <= containerSize) {

				return vec4(chessBoard(hitPos.xz), 1);
			}

			return vec4(texture(skybox, direction).rgb, 1);
		}

		else if (environmentMode == ENVIRONMENT_CHESS_BOARD) {
			if (tHitGround > 0) {
				vec3 fragToLight = normalize(lightPos - hitPos);
				return vec4(chessBoard(hitPos.xz), 1) * (0.1 + 0.5 * dot(vec3(0, 1, 0), fragToLight));
			}
			else {
				return vec4(vec3(0), 1);
			}
		}
	}
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
	return intensity2 * dot(fragToLight,normal);
}


vec4 rayTraceEnvironmentSpotLight(vec3 cameraPos, vec3 direction, vec3 lightPos) {
	float tHitGround = cameraPos.y / -direction.y;
	vec3 hitPos = cameraPos + tHitGround * direction;

	vec3 fragToLight = normalize(lightPos - hitPos);
	vec3 fragToCamera = -direction;

	vec3 normal = vec3(0, 1, 0);

	vec3 ambient = vec3(0.02);
	vec3 diffuse = vec3(0.5) * spotLightIntensity(fragToLight, normal);

	vec3 color = ambient + diffuse;

	return vec4(color, 1);
}