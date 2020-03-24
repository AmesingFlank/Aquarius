
#define M_PI 3.1415926535897932384626433832795

in vec2 TexCoord;

uniform sampler2D depthTexture;

uniform float windowWidth;
uniform float windowHeight;

uniform int smoothRadiusX;
uniform int smoothRadiusY;


uniform float sigma_d;
uniform float sigma_r;

layout (location = 0) out vec4 smoothedDepthOutput;


float bilateralFilter(ivec2 coords) {
	float centerValue = texelFetch(depthTexture, coords, 0).r;

	float valueSum = 0;
	float weightSum = 0;

	for (int dx = -smoothRadiusX; dx <= smoothRadiusX; dx++){
		for (int dy = -smoothRadiusY; dy <= smoothRadiusY; dy++) {

			float thisValue = texelFetch(depthTexture, coords + ivec2(dx, dy), 0).r;

			if (thisValue >= 0) {
				//invalid depth;
				continue;
			}

			float g = exp(-(dx * dx + dy * dy) / (2 * sigma_d * sigma_d));

			float valueDiff = thisValue - centerValue;

			float f = exp(-(valueDiff * valueDiff) / (2 * sigma_r * sigma_r));

			float thisWeight = g * f;

			valueSum += thisWeight * thisValue;
			weightSum += thisWeight;
		}
	}

	return valueSum/weightSum;
}



void main() {
	float depth = texture (depthTexture,TexCoord).r;


	if(depth>=0) {
		smoothedDepthOutput.r = depth;
		return;
	}

	vec2 xBias = vec2(1.0 / windowWidth, 0);
	vec2 yBias = vec2(0, 1.0 / windowHeight);

	ivec2 TexCoordPixels = ivec2(TexCoord * vec2(windowWidth, windowHeight));

	float result = bilateralFilter(TexCoordPixels);


	smoothedDepthOutput.r = result;

}
