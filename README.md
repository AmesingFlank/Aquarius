# Aquarius

A Fluid Simulator on GPU (CUDA)

This is Dunfan Lu's (ongoing) 3rd year project at the CS department of Oxford.

## Algorithms implemented:
* Semi-Lagrangian advection
* FLIP (Fluid Implicit Particle) advection
* PCISPH (Predicative-Corrective Incompressible Smoothed Particle Hydrodynamics)
* Position Based Dynamics
* Jacobi Pressure Solver (The projects also includes ICPCG & AMGPCG, implemented by NVIDIA)
* Screen Space Fluid Rendering
* Surface reconstruction with Marching Cubes

## demo
![](demo/3d_flip_50.png)
*3D FLIP ball-drop simulation. 50^3 grid, 8 particles per cell, mraching cubes*

![](demo/3d_flip_50.gif)
*3D FLIP ball-drop simulation. 50^3 grid, 8 particles per cell, marching cubes. Simulation + Rendering at around 20FPS on a GTX 1080 Ti*

![](demo/recording_0.gif)
*2D semi-Lagrangian*

## dependencies
* GLFW
* CUDA 
* Assimp
* STB image
* View the vcproj file for details. CMakeList doens't work now.

