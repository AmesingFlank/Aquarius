//
// Created by AmesingFlank on 2019-07-01.
//

#ifndef AQUARIUS_FLUID_2D_PCISPH_CUH
#define AQUARIUS_FLUID_2D_PCISPH_CUH

class Particle{
    float3 position;
    float3 velocity;
};


__globall__ void spatialHashing (Particle* particle, int* cellStart, int* cellEnd){

}


class Fluid_2D_PCISPH{
public:
    int particleCount;
    int gridSizeX, gridSizeY;
    float cellPhysicalSize;
    int * cellStart;
    int * cellEnd;

};

#endif //AQUARIUS_FLUID_2D_PCISPH_CUH
