
#ifndef AQUARIUS_MAC_GRID_3D_CUH
#define AQUARIUS_MAC_GRID_3D_CUH

#include <stdlib.h>
#include <memory>
#include "../Common/GpuCommons.h"
#include <cmath>
#include <vector>
#include "WeightKernels.cuh"
#include <thrust/functional.h>
#include "VolumeData.cuh"


#define CONTENT_AIR  0
#define CONTENT_FLUID  1
#define CONTENT_SOLID  2


#define get3D(arr,x,y,z) arr[(x)*(sizeY+1)*(sizeZ+1)+(y)*(sizeZ+1)+(z)]


__host__ __device__
struct VolumeCollection {
	VolumeData content;
	VolumeData pressure;
	VolumeData fluidIndex;
	VolumeData divergence;
	VolumeData particleCount;

	VolumeData velocityAccumWeight;
	VolumeData hasVelocity;

	VolumeData velocity;
	VolumeData newVelocity;

	VolumeData volumeFractions;
	VolumeData newVolumeFractions;
};

class MAC_Grid_3D{
public:
    const int sizeX;
    const int sizeY;
    const int sizeZ;
    const int cellCount;

    const float cellPhysicalSize;
    const float physicalSizeX;
    const float physicalSizeY;

	dim3 cudaGridSize;
	dim3 cudaBlockSize = dim3(8,8,8);

    int fluidCount = 0;


	MAC_Grid_3D(int X, int Y, int Z, float cellPhysicalSize_);


	VolumeCollection volumes;
	

	template<typename T>
	__device__ 
	static T getInterpolatedValueAtPoint(float x, float y, float z, int sizeX, int sizeY, int sizeZ, VolumeData volume) {

		x = max(min(x, sizeX - 1.f), 0.f);
		y = max(min(y, sizeY - 1.f), 0.f);
		z = max(min(z, sizeZ - 1.f), 0.f);


		int i = floor(x);
		int j = floor(y);
		int k = floor(z);

		float tx = x - (float)i;
		float ty = y - (float)j;
		float tz = z - (float)k;

		T x0y0 = 
			lerp(volume.readSurface<T>(i, j, k), volume.readSurface<T>(i, j, k + 1), tz);
		T x0y1 =
			lerp(volume.readSurface<T>(i, j+1, k), volume.readSurface<T>(i, j+1, k + 1), tz);
		T x1y0 =
			lerp(volume.readSurface<T>(i+1, j, k), volume.readSurface<T>(i+1, j, k + 1), tz);
		T x1y1 =
			lerp(volume.readSurface<T>(i+1, j + 1, k), volume.readSurface<T>(i+1, j + 1, k + 1), tz);

		T x0 = lerp(x0y0, x0y1, ty);
		T x1 = lerp(x1y0, x1y1, ty);

		T result = lerp(x0, x1, tx);
		return result;
	}


	__device__ 
	static float3 getPointVelocity(float3 physicalPos, float cellPhysicalSize, int sizeX, int sizeY, float sizeZ,  VolumeCollection volume){
		float x = physicalPos.x / cellPhysicalSize;
		float y = physicalPos.y / cellPhysicalSize;
		float z = physicalPos.z / cellPhysicalSize;

		float3 result;

		result.x = getInterpolatedValueAtPoint<float4>(x, y - 0.5, z - 0.5, sizeX, sizeY, sizeZ, volume.velocity).x;
		result.y = getInterpolatedValueAtPoint<float4>(x - 0.5, y, z - 0.5, sizeX, sizeY, sizeZ, volume.velocity).y;
		result.z = getInterpolatedValueAtPoint<float4>(x - 0.5, y - 0.5, z, sizeX, sizeY, sizeZ, volume.velocity).z;

		return result;
	}



	__device__ 
	static float3 getPointNewVelocity(float3 physicalPos, float cellPhysicalSize, int sizeX, int sizeY, float sizeZ,  VolumeCollection volume) {
		float x = physicalPos.x / cellPhysicalSize;
		float y = physicalPos.y / cellPhysicalSize;
		float z = physicalPos.z / cellPhysicalSize;

		float3 result;

		result.x = getInterpolatedValueAtPoint<float4>(x, y - 0.5, z - 0.5, sizeX, sizeY, sizeZ, volume.newVelocity).x;
		result.y = getInterpolatedValueAtPoint<float4>(x - 0.5, y, z - 0.5, sizeX, sizeY, sizeZ, volume.newVelocity).y;
		result.z = getInterpolatedValueAtPoint<float4>(x - 0.5, y - 0.5, z, sizeX, sizeY, sizeZ, volume.newVelocity).z;

		return result;
	}




	__device__ __host__
	static float3 getPhysicalPos(int x, int y, int z, float cellPhysicalSize) {
		return make_float3((x + 0.5f) * cellPhysicalSize, (y + 0.5f) * cellPhysicalSize, (z + 0.5f) * cellPhysicalSize);
	}



	void commitContentChanges();

	void updateFluidCount();


	float getMaxSpeed();

};

#endif //AQUARIUS_MAC_GRID_3D_CUH
