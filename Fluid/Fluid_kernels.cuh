#include "../Common/GpuCommons.h"

__global__  inline void findCellStartEndImpl(int* particleHashes,
	int* cellStart, int* cellEnd,
	int particleCount) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= particleCount) return;

	int thisHash = particleHashes[index];

	

	if (index == 0 || particleHashes[index - 1] < thisHash) {
		cellStart[thisHash] = index;
	}

	if (index == particleCount - 1 || particleHashes[index + 1] > thisHash) {
		cellEnd[thisHash] = index;
	}
	
}
