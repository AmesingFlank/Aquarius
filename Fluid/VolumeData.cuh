#pragma once

#include "../Common/GpuCommons.h"

__host__ __device__
struct VolumeData {
	cudaArray* array;
	cudaTextureObject_t texture;
	cudaSurfaceObject_t surface;

	
	template<typename T>
	__device__
	T readSurface(int x, int y, int z) {
		return surf3Dread<T>(surface, x * sizeof(T), y, z);
	}

	template<typename T>
	__device__
	void writeSurface(T value,int x,int y,int z){
		surf3Dwrite<T>(value, surface, x * sizeof(T), y, z);
	}

	template<typename T>
	__device__
	T readTexture(float x, float y, float z) {
		return tex3D<T>(texture, x , y, z);
	}
};

template<typename T>
inline
__global__
void clearField3D(VolumeData field, int sizeX, int sizeY, int sizeZ,T valueToClear) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= sizeX || y >= sizeY || z >= sizeZ) return;
	surf3Dwrite<T>(valueToClear, field.surface, x * sizeof(T), y, z);
};


template<typename T>
inline
VolumeData createField3D(int sizeX, int sizeY, int sizeZ, dim3 cudaGridSize, dim3 cudaBlockSize,T initialValue, bool filter, cudaTextureAddressMode addressMode = cudaAddressModeClamp) {
	VolumeData result;

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
	cudaExtent extent = { (size_t)sizeX ,(size_t)sizeY ,(size_t)sizeZ };
	HANDLE_ERROR(cudaMalloc3DArray(&result.array, &channelDesc, extent, cudaArraySurfaceLoadStore));

	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = result.array;

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	
	texDesc.addressMode[0] = addressMode;
	if (filter)
		texDesc.filterMode = cudaFilterModeLinear;
	else
		texDesc.filterMode = cudaFilterModePoint;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;

	HANDLE_ERROR(cudaCreateTextureObject(&result.texture, &resDesc, &texDesc, nullptr));

	HANDLE_ERROR(cudaCreateSurfaceObject(&result.surface, &resDesc));

	clearField3D<T> << <cudaGridSize, cudaBlockSize >> > (result, sizeX, sizeY, sizeZ,initialValue);
	CHECK_CUDA_ERROR("create field");
	return result;
};



inline void releaseField3D(VolumeData volume) {
	HANDLE_ERROR(cudaFreeArray(volume.array));
	HANDLE_ERROR(cudaDestroySurfaceObject(volume.surface));
	HANDLE_ERROR(cudaDestroyTextureObject(volume.texture));
}