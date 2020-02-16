
#include "../../Common/GpuCommons.h"

namespace InkRendererParticle {
	__device__ __host__ struct Particle {
		float3 position = make_float3(0, 0, 0);
		float3 velocity = make_float3(0, 0, 0);

		__device__ __host__
			Particle() {

		}
		Particle(float3 pos) :position(pos) {

		}
	};

	struct Renderer {
		Particle* particles;

		int VBO;
		int VAO;

		float* positionsDevice;

	};
}