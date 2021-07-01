cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(Aquarius LANGUAGES CXX CUDA)


add_subdirectory(Dependencies/glfw)
include_directories(Dependencies/glfw/include)

add_subdirectory(Dependencies/glew/build/cmake)
include_directories(Dependencies/glew/include)

include_directories(Dependencies/include)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})


set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math")

set(CMAKE_CXX_STANDARD 14)



add_executable(Aquarius 
 Fluid/Fluid_3D_common.cu
 Fluid/Fluid_3D_FLIP.cu
 Fluid/Fluid_3D_kernels.cu
 Fluid/Fluid_3D_PBF.cu
 Fluid/Fluid_3D_PCISPH.cu
 Fluid/MAC_Grid_3D.cu
 main.cu
 Rendering/Renderer3D/Container.cu
 Rendering/Renderer3D/FluidMeshRenderer.cu
 Rendering/Renderer3D/Mesher.cu
 Rendering/Renderer3D/PointSprites.cu
 Rendering/Renderer3D/ScreenSpaceNormal.cu
 Rendering/Renderer3D/Skybox.cu
 Rendering/Shader.cpp
 UI/ui.cpp

  Fluid/FluidConfig.cuh
 Fluid/Fluid_3D.cuh
 Fluid/Fluid_3D_common.cuh
 Fluid/Fluid_3D_FLIP.cuh
 Fluid/Fluid_3D_kernels.cuh
 Fluid/Fluid_3D_PBF.cuh
 Fluid/Fluid_3D_PCISPH.cuh
 Fluid/Fluid_kernels.cuh
 Fluid/MAC_Grid_3D.cuh
 Fluid/VolumeData.cuh
 Fluid/WeightKernels.cuh
 Rendering/Renderer3D/FluidMeshRenderer.cuh
 Rendering/Renderer3D/Mesher.cuh
 Rendering/Renderer3D/ScreenSpaceNormal.cuh

 Common/Commons.h
 Common/GpuCommons.h
 Common/InputHandler.h
 Fluid/Fluid.h
 Rendering/DrawCommand.h
 Rendering/Quad.h
 Rendering/Renderer3D/camera.h
 Rendering/Renderer3D/Container.h
 Rendering/Renderer3D/PointSprites.h
 Rendering/Renderer3D/Skybox.h
 Rendering/Shader.h
 Rendering/WindowInfo.h
 UI/ui.h

)

IF(APPLE)
set_property(TARGET Aquarius
        PROPERTY
        BUILD_RPATH /usr/local/cuda/lib)
ENDIF(APPLE)

target_link_libraries(Aquarius glfw ${GLFW_LIBRARIES})
target_link_libraries(Aquarius glew_s ${GLEW_LIBRARIES})

add_custom_command(TARGET Aquarius PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/resources/ $<TARGET_FILE_DIR:Aquarius>/resources)

set_property(TARGET Aquarius PROPERTY CUDA_STANDARD 14)

