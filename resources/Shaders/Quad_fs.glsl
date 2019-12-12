#version 330

in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D quadTexture;

void main() {
    FragColor = texture (quadTexture,TexCoord);
    //FragColor = vec4(TexCoord,0,1);
}
