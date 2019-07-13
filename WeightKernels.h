//
// Created by AmesingFlank on 2019-07-03.
//

#ifndef AQUARIUS_WEIGHTKERNELS_H
#define AQUARIUS_WEIGHTKERNELS_H

#include <math.h>

__host__ __device__
float poly6(float3 v,float h){
    float r = length(v);
    if(0<=r && r<=h){
        return pow(h*h-r*r,3)*315/(64*M_PI*pow(h,9));
    }
    else{
        return 0;
    }
}

__host__ __device__
float poly6(float2 v,float h){
    float r = length(v);
    if(0<=r && r<=h){
        return pow(h*h-r*r,3)*4/(M_PI*pow(h,8));
    }
    else{
        return 0;
    }
}

// the grad/laplacian of the poly6 are not used, due to the kernel having 0 gradient at center

__host__ __device__
float3 poly6_grad(float3 v,float h){
    float r = length(v);
    if(0<=r && r<=h){
        return v* (pow(h*h-r*r,2)*945/(-32*M_PI*pow(h,9)) );
    }
    else{
        return make_float3(0,0,0);
    }
}

__host__ __device__
float poly6_laplacian(float3 v,float h){
    float r = length(v);
    if(0<=r && r<=h){
        float h2_r2 = (h*h-r*r);
        return h2_r2 * (r*r - h2_r2*3.f/4.f) * 945 / (8*M_PI*pow(h,9));
    }
    else{
        return 0;
    }
}


__host__ __device__
float2 poly6_grad(float2 v,float h){
    float r = length(v);
    if(0<=r && r<=h){
        return v* (pow(h*h-r*r,2)*24/(M_PI*pow(h,8)) );
    }
    else{
        return make_float2(0,0);
    }
}

__host__ __device__
float spikey(float3 v,float h){
    float r = length(v);
    if(0<=r && r<=h){
        return pow(h-r,3)*15/(M_PI*pow(h,6));
    }
    else{
        return 0;
    }
}

__host__ __device__
float spikey(float2 v,float h){
    float r = length(v);
    if(0<=r && r<=h){
        return pow(h-r,3)*10/(M_PI*pow(h,5));
    }
    else{
        return 0;
    }
}

__host__ __device__
float3 spikey_grad(float3 v,float h){
    float r = length(v);
    float3 v_unit = v;
    if(r!=0){
        v_unit/=r;
    }
    if(0<=r && r<=h){
        return -v_unit * pow(h-r,2)*45/(M_PI*pow(h,6));
    }
    else{
        return make_float3(0,0,0);
    }
}

__host__ __device__
float2 spikey_grad(float2 v,float h){
    float r = length(v);
    float2 v_unit = v;
    if(r!=0){
        v_unit/=r;
    }
    if(0<=r && r<=h){
        return -v_unit*pow(h-r,2)*30/(M_PI*pow(h,5));
    }
    else{
        return make_float2(0,0);
    }
}


// unused if solving Euler equation. Used of N-S equation

__host__ __device__
float viscosity(float3 v,float h){
    float r = length(v);
    if(0<=r && r<=h){
        return ( - pow(r,3)/(2*pow(h,3)) + r*r/(h*h) + h/(2*r) - 1) * 15 / (2*M_PI*pow(h,3));
    }
    else{
        return 0;
    }
}


__host__ __device__
float viscosity(float2 v,float h){
    float r = length(v);
    if(0<=r && r<=h){
        return ( - 4*pow(r,3)/(9*pow(h,3)) + r*r/(h*h) ) * 90 / (29*M_PI*h*h);
    }
    else{
        return 0;
    }
}

__host__ __device__
float viscosity_laplacian(float3 v,float h){
    float r = length(v);
    if(0<=r && r<=h){
        return (1-r/h)*45/(M_PI*h*h);
    }
    else{
        return 0;
    }
}


__host__ __device__
float viscosity_laplacian(float2 v,float h){
    float r = length(v);
    if(0<=r && r<=h){
        return (h-r)*360 / (29*M_PI*pow(h,5));
    }
    else{
        return 0;
    }
}





#endif //AQUARIUS_WEIGHTKERNELS_H
