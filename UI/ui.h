#pragma once
#include "../Common/GpuCommons.h"

#include "../Fluid/FluidConfig.cuh"
#include <functional>

struct nk_context;

nk_context* createUI(GLFWwindow* win);

void drawUI(nk_context* ctx, FluidConfig& fluidConfig, std::function<void()> onStart);