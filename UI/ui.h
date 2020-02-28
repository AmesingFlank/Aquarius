#pragma once
#include "../Common/GpuCommons.h"

struct nk_context;

nk_context* createUI(GLFWwindow* win);

void drawUI(nk_context* ctx);