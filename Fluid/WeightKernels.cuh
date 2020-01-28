
#ifndef AQUARIUS_WEIGHTKERNELS_H
#define AQUARIUS_WEIGHTKERNELS_H

#define _USE_MATH_DEFINES
#include <math.h>

#define PI (3.14159265358979323846f)



__host__ __device__
inline
float poly6(float3 v, float h2,float h9) {
	float r2 = dot(v, v);
	if (r2 <= h2) {

		float h2_r2 = h2 - r2;
		float h2_r2_3 = h2_r2 * h2_r2 * h2_r2;

		return h2_r2_3 * 315.f / (64.f * PI * h9);
	}
	else {
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



__host__ __device__
inline
float3 spikey_grad(float3 v,float h,float h6){
    float r = length(v);
    float3 v_unit = v;
    if(r!=0){
        v_unit/=r;
    }
    if(0<=r && r<=h){
		
        return -v_unit *(h-r) * (h-r)*45.f/(PI*h6);
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
	float r2 = dot(r, r);
	float h2 = support * support;
	float temp = 1.f - r2 / h2 ;
	return max(0.f,temp*temp*temp);
}

__device__ __host__
float inline dunfanKernel(float3 r, float support) {
	float r2 = dot(r, r);
	float h2 = support * support;
	if (r2 > h2) return 0;
	float temp = 1.f - r2 / (h2 * 3.f);
	return max(0.f, temp * temp * temp);
}

__device__ __host__
float inline pcaKernel(float3 r, float support) {
	return max(0.f,1.f - pow(length(r) / support, 3.f));
}

__device__  __host__
float static inline cubic_spline_kernel(const float r, const float radius,const float radius3)
{
	const auto q = 2.0f * r / radius;

	if (q > 2.0f ) return 0.0f;
	else {
		const float a = 0.25f / (PI * radius3);
		return a * ((q > 1.0f) ? (2.0f - q) * (2.0f - q) * (2.0f - q) : ((3.0f * q - 6.0f) * q * q + 4.0f));
	}
}

__device__  __host__
float3 static inline cubic_spline_kernel_gradient(const float3 v, const float vLength, const float radius,const float radius5)
{

	const auto q = 2.0f * vLength / radius;

	if (q > 2.0f) return make_float3(0.0f);
	else {

		const float3 a = v / (PI * (q + 1e-6) * radius5);
		return a * ((q > 1.0f) ? ((12.0f - 3.0f * q) * q - 12.0f) : ((9.0f * q - 12.0f) * q));
	}
}

#undef PI


#endif //AQUARIUS_WEIGHTKERNELS_H
