#include<string>
static const std::string PointSprites_fs = R"(

#version 330 core
in vec3 TexCoords;
out vec4 color;

in vec3 debug;

void main()
{  
	vec3 N;
	N.xy = gl_PointCoord* 2.0 - vec2(1.0); 
	float mag = dot(N.xy, N.xy);
	if(mag > 1.0) 
     discard; // kill pixels outside circle
	N.z = sqrt(1.0-mag);


	// calculate lighting
	vec3 lightDir = vec3(0,1,0);

	float diffuse = max(0.0, dot(lightDir, N));
	color = vec4(vec3(0,0,1) * diffuse,1);

  
    //color = vec4(gl_PointCoord,0,1);
	//color = vec4(debug , 1);
}

)";
