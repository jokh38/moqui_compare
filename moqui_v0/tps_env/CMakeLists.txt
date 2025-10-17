cmake_minimum_required(VERSION 3.10)
project(env LANGUAGES CXX CUDA)

# RTX A5000 = sm_86
set(CMAKE_CUDA_ARCHITECTURES 86)
# 런타임을 동적 링크로 (ldd에 libcudart가 보이도록)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

set(GPU ON)

# The extension of the main code should be cpp to compile it using g++
# for CPU version and using nvcc for GPU version.
# It will not be compiled using g++ if the extension is cu

# setting Cuda 11.7 
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.7/bin/nvcc CACHE FILEPATH "CUDA 11.7 nvcc" FORCE)
set(CUDATOOLkit_ROOT /usr/local/cuda-11.7 CACHE PATH "CUDA 11.7 root" FORCE)
add_executable(tps_env tps_env.cpp)

if (GPU)
    message("Compile using nvcc")
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)
    set_property(TARGET tps_env PROPERTY CUDA_ARCHITECTURES 86)
    set(CMAKE_CUDA_COMPILER nvcc)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w --use_fast_math")
    set_source_files_properties(tps_env.cpp PROPERTIES LANGUAGE CUDA)
else ()
    message("Compile using g++")
    enable_language(CXX)
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS ON)
    set(CMAKE_CXX_COMPILER g++)
endif ()

find_package(GDCM REQUIRED)
include(${GDCM_USE_FILE})

if (APPLE)
    find_library(COREFOUNDATION_LIBRARY CoreFoundation)
endif ()

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${GDCM_INCLUDE_DIRS}
        ../
        ../moqui
        ${CUDAToolkit_INCLUDE_DIRS}
        /usr/local/cuda/include
)

link_directories(${GDCM_DIR})

target_link_libraries(tps_env PRIVATE
        CUDA::cudart
        ${COREFOUNDATION_LIBRARY}
        gdcmCommon
        gdcmDSED
        gdcmMEXD
        gdcmjpeg12
        gdcmjpeg8
        gdcmDICT
        gdcmIOD
        gdcmMSFF
        gdcmjpeg16
        )

