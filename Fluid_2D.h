//
// Created by AmesingFlank on 2019-07-04.
//

#ifndef AQUARIUS_FLUID_2D_H
#define AQUARIUS_FLUID_2D_H

#include <vector>
#include <utility>
#include "CudaCommons.h"
#include <unordered_map>


class Fluid_2D{
public:
    GLuint texture;
    void initTexture(){
        glGenTextures(1,&texture);
        glBindTexture(GL_TEXTURE_2D,texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    Fluid_2D(){
        initTexture();
    }
    virtual void updateTexture() = 0;
};

#endif //AQUARIUS_FLUID_2D_H
