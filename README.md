# Aquarius

A Fluid Simulator on GPU (CUDA)

This is Candidate 1023011's 3rd year project at the CS department of Oxford.

## Algorithms implemented:
* FLIP (Fluid Implicit Particle) advection
* PCISPH (Predicative-Corrective Incompressible Smoothed Particle Hydrodynamics)
* Position Based Fluid
* Jacobi Pressure Solver 
* Surface reconstruction with Marching Cubes
* Multiphase Fluid Diffusion

## demo
![](demo/3d_flip_50.png)
*3D FLIP ball-drop simulation. 50^3 grid, 8 particles per cell, mraching cubes*

![](demo/3d_flip_50.gif)
*3D FLIP ball-drop simulation. 50^3 grid, 8 particles per cell, marching cubes. Simulation + Rendering at around 20FPS on a GTX 1080 Ti*


Multiphase Diffusion       |  A red ball of fluid droppping into a blue box of fluid, with diffusion
:-------------------------:|:-------------------------:
![](demo/multiphase_ball_0_0.png)  |  ![](demo/multiphase_ball_0_1.png)
![](demo/multiphase_ball_0_2.png)  |  ![](demo/multiphase_ball_0_3.png)




![](demo/recording_0.gif)
*2D semi-Lagrangian. Implemented in the early phases of the project*

## Dependencies
* GLFW
* GLEW
* glm
* STB image
* Nukclear

The cmake file should be able to handle these automatically.

## Credits
The project uses some of the cubemaps from http://www.humus.name as background.

## More Images

![](report/images/FrontSinglephase_cropped.png)
![](report/images/FrontMultiphase_cropped.png)


Ball Drop       |  Single Phase FLIP
:-------------------------:|:-------------------------:
![](report/images/balldrop_cropped2/single0.png)  |  ![](report/images/balldrop_cropped2/single1.png)
![](report/images/balldrop_cropped2/single2.png)  |  ![](report/images/balldrop_cropped2/single3.png)

Ball Drop       |  Multi Phase FLIP
:-------------------------:|:-------------------------:
![](report/images/balldrop_cropped2/multi0.png)  |  ![](report/images/balldrop_cropped2/multi1.png)
![](report/images/balldrop_cropped2/multi2.png)  |  ![](report/images/balldrop_cropped2/multi3.png)


Ball Drop       |  Single Phase PBF
:-------------------------:|:-------------------------:
![](report/images/balldrop_cropped2/pbf0.png)  |  ![](report/images/balldrop_cropped2/pbf1.png)
![](report/images/balldrop_cropped2/pbf2.png)  |  ![](report/images/balldrop_cropped2/pbf3.png)

![](report/logoF.png)

![](report/logoE.png)