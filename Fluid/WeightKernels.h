
#ifndef AQUARIUS_WEIGHTKERNELS_H
#define AQUARIUS_WEIGHTKERNELS_H

#define _USE_MATH_DEFINES
#include <math.h>

__host__ __device__  
inline
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
inline
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
inline
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
inline
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
inline
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
inline
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
inline
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
inline
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
inline
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
inline
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
inline
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
inline
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
inline
float viscosity_laplacian(float2 v,float h){
    float r = length(v);
    if(0<=r && r<=h){
        return (h-r)*360 / (29*M_PI*pow(h,5));
    }
    else{
        return 0;
    }
}



__device__ __host__
float inline trilinearUnitKernel(float r){
    r=abs(r);
    if(r>1) return 0;
    return 1-r;
}

__device__ __host__
float inline trilinearHatKernel(float2 r,float support){
    return trilinearUnitKernel(r.x / support) * trilinearUnitKernel(r.y / support);
}

__device__ __host__
float inline trilinearHatKernel(float3 r, float support) {
	return trilinearUnitKernel(r.x / support) * trilinearUnitKernel(r.y / support) * trilinearUnitKernel(r.z / support);
}


__device__ __host__
float inline quadraticBSplineUnitKernel(float r){
    if(-3.0/2.0 <= r && r <= -1.0/2.0){
        return pow(r+3.0/2.0,2)/2;
    }
    if(-1.0/2.0 < r && r < 1.0/2.0){
        return 3.0/4.0 - r*r;
    }
    if(1.0/2.0 <= r && r <= 3.0/2.0 ){
        return pow(-r+3.0/2.0,2)/2;
    }
    return 0;
}

__device__ __host__
float inline quadraticBSplineKernel(float2 r,float support){
    return quadraticBSplineUnitKernel(r.x / support) * quadraticBSplineUnitKernel(r.y / support);
}

__device__ __host__
float inline quadraticBSplineKernel(float3 r, float support) {
	return quadraticBSplineUnitKernel(r.x / support) * quadraticBSplineUnitKernel(r.y / support) * quadraticBSplineUnitKernel(r.z / support);
}


__device__ __host__
float inline zhu05Kernel(float3 r, float support) {
	float s = length(r) / support;
	return max(0.0,pow((1.0-s*s),3));
}

__device__ __host__
float inline pcaKernel(float3 r, float support) {
	return 1 - pow(length(r) / support, 3);
}

__device__ __host__
float inline Bcubic(float3 r, float support) {
	float s = length(r) / support;
	if (s < 1) {
		return (1 - s * s * 3.f / 2.f + s * s * s * 3.f / 4.f) / M_PI;
	}
	else if (s < 2) {
		return pow(2 - s, 3) / (4.0 * M_PI);
	}
	return 0;
}

#endif //AQUARIUS_WEIGHTKERNELS_H
