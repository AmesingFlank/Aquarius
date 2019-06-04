#include<string>
static const std::string Quad_vs = R"(

#version 330

layout (location = 10) in vec3 vPos;
layout (location = 15) in vec2 texCoord;

out vec2 TexCoord;

void main() {
    gl_Position = vec4(vPos,1);
    TexCoord=texCoord;
}

)";
