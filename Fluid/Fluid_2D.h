//
// Created by AmesingFlank on 2019-07-04.
//

#ifndef AQUARIUS_FLUID_2D_H
#define AQUARIUS_FLUID_2D_H

#include <vector>
#include <utility>
#include "../GpuCommons.h"
#include <unordered_map>
#include "Fluid.h"
#include "../Rendering/Quad.h"



class Fluid_2D : Fluid{
public:
	Quad quad;
    GLuint texture;
	unsigned char* imageGPU = nullptr;
	unsigned char* imageCPU = nullptr;
	int imageSizeX;
	int imageSizeY;
	int imageMemorySize;

	Fluid_2D()
	{
		initTexture();
	}

    void initTexture(){
		
        glGenTextures(1,&texture);
        glBindTexture(GL_TEXTURE_2D,texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

	void initTextureImage(int imageSizeX_, int imageSizeY_) {
		releaseImageMemory();

		imageSizeX = imageSizeX_;
		imageSizeY = imageSizeY_;
		imageMemorySize = imageSizeX * imageSizeY * 4;

		allocateImageMemory();
	}

	void allocateImageMemory() {
		imageCPU = (unsigned char*)malloc(imageMemorySize);
		HANDLE_ERROR(cudaMalloc(&imageGPU, imageMemorySize));
	}

	void releaseImageMemory() {
		if (imageGPU) {
			HANDLE_ERROR(cudaFree(imageGPU));
			imageGPU = nullptr;
		}
		if (imageCPU) {
			free(imageCPU);
			imageCPU = nullptr;
		}
	}

   

	void drawImage() {
		HANDLE_ERROR(cudaMemcpy(imageCPU, imageGPU, imageMemorySize, cudaMemcpyDeviceToHost));

		glBindTexture(GL_TEXTURE_2D, texture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, imageSizeX,imageSizeY, 0, GL_RGBA, GL_UNSIGNED_BYTE, imageCPU);
		glGenerateMipmap(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, 0);
		printGLError();

		quad.draw(texture);
		printGLError();
	}

	virtual ~Fluid_2D() {
		releaseImageMemory();
	}

};

#endif //AQUARIUS_FLUID_2D_H
