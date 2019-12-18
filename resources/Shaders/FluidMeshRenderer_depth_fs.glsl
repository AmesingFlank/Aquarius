#version 330 core
out vec4 depthOutput;
in vec3 fragPosViewSpace;
void main()
{
	depthOutput.r = fragPosViewSpace.z;
	depthOutput.a = 1;
}
