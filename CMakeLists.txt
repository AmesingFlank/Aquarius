cmake_minimum_required(VERSION 3.13)
project(Aquarius)

set(CMAKE_CXX_STANDARD 17)

add_executable(Aquarius
        main.cu GpuCommons.h MAC_Grid_2D.cuh Fluid_2D_SemiLagrange.cuh SPD_Solver.h Rendering/Shader.h Rendering/Quad.h Fluid_2D_PCISPH.cuh WeightKernels.h Fluid_2D.h Fluid_2D_PCISPH_CPU.h Fluid_2D_PoistionBased_CPU.h Fluid_2D_FLIP.cuh MAC_Grid_3D.cuh Rendering/Renderer3D/Renderer3D.h Rendering/Renderer3D/Skybox.h Rendering/Renderer3D/camera.h InputHandler.h )

set(OPTM_LEVEL "-O0")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${OPTM_LEVEL}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPTM_LEVEL}")



enable_language(CUDA)

set_property(TARGET Aquarius PROPERTY CUDA_STANDARD 11)

set_property(TARGET Aquarius
        PROPERTY CUDA_SEPARABLE_COMPILATION ON)

set_property(TARGET Aquarius
        PROPERTY
        BUILD_RPATH /usr/local/cuda/lib)

target_include_directories(Aquarius PRIVATE /usr/local/cuda/samples/common/inc/)

target_link_libraries(Aquarius -lcublas -lcurand -lcusparse -lcusolver)

target_link_libraries(Aquarius -lglfw3 -lglew)

target_link_libraries(Aquarius "-framework OpenGL")




#AMGX stuff
target_include_directories(Aquarius PRIVATE /Users/AmesingFlank/Downloads/AMGX-master/built/include_all_frank/include)
target_link_libraries(Aquarius /Users/AmesingFlank/Downloads/AMGX-master/built/core/libamgx_core.a)
target_link_libraries(Aquarius /Users/AmesingFlank/Downloads/AMGX-master/built/base/libamgx_base.a)
target_link_libraries(Aquarius /Users/AmesingFlank/Downloads/AMGX-master/built/template_plugin/libamgx_template_plugin.a)
target_link_libraries(Aquarius /Users/AmesingFlank/Downloads/AMGX-master/built/eigensolvers/libamgx_eigensolvers.a)

add_custom_command(TARGET Aquarius PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/resources/ $<TARGET_FILE_DIR:Aquarius>/resources)
