
#define M_PI 3.1415926535897932384626433832795

in vec2 TexCoord;

uniform sampler2D depthTexture;
uniform float windowWidth;
uniform float windowHeight;
uniform float zoom;

layout (location = 0) out vec4 normalOutput;

void main() {
	float depth = texture (depthTexture,TexCoord).r;
	

	if (depth >= 0) {
		discard; return;
	}


	vec2 xBias = vec2(1.0/windowWidth,0);
	vec2 yBias = vec2(0,1.0/windowHeight);

	ivec2 TexCoordPixels = ivec2(TexCoord * vec2(windowWidth, windowHeight));

	float depthLeft = texelFetch(depthTexture, TexCoordPixels - ivec2(1, 0),0).r;
	float depthRight = texelFetch(depthTexture, TexCoordPixels + ivec2(1, 0), 0).r;
	float depthUp = texelFetch(depthTexture, TexCoordPixels + ivec2(0, 1), 0).r;
	float depthDown = texelFetch(depthTexture, TexCoordPixels - ivec2(0, 1), 0).r;

	float dzdxLeft = depth - depthLeft;
	float dzdxRight = depthRight - depth;
	float dzdyUp = depthUp - depth;
	float dzdyDown = depth - depthDown;

	float dzdx = dzdxLeft;
	if( abs(dzdx) > abs(dzdxRight)){
		dzdx = dzdxRight;
	}

	float dzdy = dzdyUp;
	if( abs(dzdy) > abs(dzdyDown)){
		dzdy = dzdyDown;
	}

	float dxViewSpace = tan(zoom*M_PI / 180) * abs(depth) * 2 / windowWidth;
	float dyViewSpace = dxViewSpace * windowHeight / windowWidth ;
		
	vec3 tangentX = vec3(dxViewSpace, 0, dzdx);
	vec3 tangentY = vec3(0,dyViewSpace, dzdy);
	vec3 normal = cross(tangentX,tangentY);
	normal = normalize(normal);
	if(normal.z < 0) normal = -normal;

	//normal = vec3(1, 1, 1);

	normalOutput = vec4(normal,1);
	

}