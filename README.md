# Aquarius

A Fluid Simulator on GPU (CUDA)

This is Dunfan Lu's (ongoing) 3rd year project at the CS department of Oxford.

## Algorithms implemented:
* Semi-Lagrangian advection
* FLIP (Fluid Implicit Particle) advection
* PCISPH (Predicative-Corrective Incompressible Smoothed Particle Hydrodynamics)
* Position Based Fluid
* Jacobi Pressure Solver 
* Screen Space Fluid Rendering
* Surface reconstruction with Marching Cubes
* Multiphase Fluid Diffusion

## demo
![](demo/3d_flip_50.png)
*3D FLIP ball-drop simulation. 50^3 grid, 8 particles per cell, mraching cubes*

![](demo/3d_flip_50.gif)
*3D FLIP ball-drop simulation. 50^3 grid, 8 particles per cell, marching cubes. Simulation + Rendering at around 20FPS on a GTX 1080 Ti*


Multiphase Diffusion       |  -
:-------------------------:|:-------------------------:
![](demo/multiphase_ball_0_0.png)  |  ![](demo/multiphase_ball_0_1.png)
![](demo/multiphase_ball_0_2.png)  |  ![](demo/multiphase_ball_0_3.png)
-       |  -
*A red ball of fluid droppping into a blue box of fluid, with diffusion*


![](demo/recording_0.gif)
*2D semi-Lagrangian. Implemented in the early phases of the project*

## dependencies
* GLFW
* CUDA 
* STB image
* View the vcproj file for details. CMakeList doens't work now.

