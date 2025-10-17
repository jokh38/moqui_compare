cmake_minimum_required(VERSION 3.10)
project(env LANGUAGES CXX)

option(GPU "Enable GPU support" OFF)

# Build tps_env executable
add_executable(tps_env tps_env.cpp)

# Link with basic libraries for now
add_dependencies(tps_env moqui_rev_lib)
target_link_libraries(tps_env PRIVATE moqui_rev_lib)

# Disable warnings as errors for tps_env targets
target_compile_options(tps_env PRIVATE -Wno-error -w)

if (GPU)
    message(STATUS "Compiling with CUDA support")
    # RTX A5000 = sm_86
    set(CMAKE_CUDA_ARCHITECTURES 86)
    # 런타임을 동적 링크로 (ldd에 libcudart가 보이도록)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

    # setting Cuda 11.7
    set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.7/bin/nvcc CACHE FILEPATH "CUDA 11.7 nvcc" FORCE)
    set(CUDATOOLkit_ROOT /usr/local/cuda-11.7 CACHE PATH "CUDA 11.7 root" FORCE)

    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)

    set_property(TARGET tps_env PROPERTY CUDA_ARCHITECTURES 86)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w --use_fast_math")
    set_source_files_properties(tps_env.cpp PROPERTIES LANGUAGE CUDA)

    include_directories(${CUDATOOLkit_INCLUDE_DIRS} /usr/local/cuda/include)
    target_link_libraries(tps_env PRIVATE CUDA::cudart)
else ()
    message(STATUS "Compiling without CUDA support (CPU only)")
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS ON)
    set(CMAKE_CXX_COMPILER g++)
    # Temporarily disable warnings as errors to get compilation working
    target_compile_options(tps_env PRIVATE -Wno-error)
endif ()

# Try to find GDCM, but make it optional for now
# First try to find GDCM in the specified location
set(GDCM_DIR "/usr/include/gdcm-3.0" CACHE PATH "Path to GDCM 3.0")
find_package(GDCM QUIET)

if(GDCM_FOUND)
    message(STATUS "GDCM found - enabling DICOM support")
    include(${GDCM_USE_FILE})
    set(GDCM_AVAILABLE TRUE)
else()
    # Manual configuration for GDCM 3.0 if find_package fails
    if(EXISTS "/usr/include/gdcm-3.0")
        message(STATUS "GDCM 3.0 found manually at /usr/include/gdcm-3.0")
        set(GDCM_AVAILABLE TRUE)
        set(GDCM_INCLUDE_DIRS "/usr/include/gdcm-3.0")
        set(GDCM_LIBRARIES "gdcmCommon;gdcmDSED;gdcmMEXD;gdcmDICT;gdcmIOD;gdcmMSFF")
    else()
        message(WARNING "GDCM not found - compiling without DICOM support")
        set(GDCM_AVAILABLE FALSE)
    endif()
endif()

if (APPLE)
    find_library(COREFOUNDATION_LIBRARY CoreFoundation)
endif ()

include_directories(
        ${PROJECT_SOURCE_DIR}
        ../
)

# Only add GDCM include directories if GDCM is available
if(GDCM_AVAILABLE)
    if(DEFINED GDCM_INCLUDE_DIRS)
        include_directories(${GDCM_INCLUDE_DIRS})
    else()
        include_directories("/usr/include/gdcm-3.0")
    endif()
endif()

# Only link GDCM libraries if available
if(GDCM_AVAILABLE)
    link_directories(${GDCM_DIR})
    target_link_libraries(tps_env PRIVATE
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
            z
            )
else()
    target_link_libraries(tps_env PRIVATE
            ${COREFOUNDATION_LIBRARY}
            )
endif()


# Try to find DCMTK
find_package(DCMTK)

if(DCMTK_FOUND)
    message(STATUS "DCMTK found - enabling DICOM RT support")
    include_directories(${DCMTK_INCLUDE_DIRS})
    target_link_libraries(tps_env PRIVATE ${DCMTK_LIBRARIES})
    # Define DCMTK_FOUND for conditional compilation
    target_compile_definitions(tps_env PRIVATE DCMTK_FOUND=1)
else()
    message(WARNING "DCMTK not found - compiling without DICOM RT support")
    message(STATUS "DICOM RT Dose output will fall back to MHD format")
    # Define that DCMTK is not found for conditional compilation
    target_compile_definitions(tps_env PRIVATE DCMTK_FOUND=0)
endif()
