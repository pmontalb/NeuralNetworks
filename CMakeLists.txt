include_guard()

cmake_minimum_required(VERSION 3.14)
project(NeuralNetworks)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-10.1/bin/nvcc CACHE PATH "" FORCE)
enable_language(CUDA)

# CudaLight
include_directories(../CudaLight/CudaLight)
include_directories(../NpyCpp/Npy++)
include_directories(../NpyCpp/MemoryMapping)
add_subdirectory(../CudaLight ${CMAKE_BINARY_DIR}/CudaLight EXCLUDE_FROM_ALL)

add_library(
        NeuralNetworkKernels
        NeuralNetworkKernels/ObjectiveFunctions.cu
)
target_include_directories(NeuralNetworkKernels PRIVATE ../CudaLightKernels)
target_include_directories(NeuralNetworkKernels PRIVATE NeuralNetworkKernels)
target_compile_options(NeuralNetworkKernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -gencode arch=compute_61,code=sm_61 -m64 -restrict --compiler-options -Wall>)
target_compile_options(NeuralNetworkKernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler -Wno-unknown-pragmas>)
target_link_libraries(NeuralNetworkKernels CudaLightKernels)

# set warnings
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(WARNINGS -Weverything -Werror
            -Wno-c++98-compat -Wno-c++98-compat-pedantic
            -Wno-reserved-id-macro -Wno-switch-enum -Wno-unknown-pragmas -Wno-disabled-macro-expansion
            -Wno-unneeded-internal-declaration -Wno-weak-vtables -Wno-unused-function -Wno-padded -Wno-unused-macros
            -Wno-exit-time-destructors)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(WARNINGS -Wall -Wextra -Werror -Wno-unknown-pragmas -Wno-unused-function)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    set(WARNINGS -Wall -Wextra -Werror -Wno-unknown-pragmas -Wfatal-errors -fstrict-aliasing -pedantic -pedantic-errors)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    message(FATAL "MSVC not supported")
endif()

set(DEBUG_FLAGS "")
if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    set(DEBUG_FLAGS "-D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC")
endif()

add_library(
        NeuralNetworks
        NeuralNetworks/NeuralNetworksManager.cpp
)
target_include_directories(NeuralNetworks PUBLIC ./)
target_include_directories(NeuralNetworks PUBLIC ./NeuralNetworks)
target_include_directories(NeuralNetworks PUBLIC ./NeuralNetworkKernels)
target_include_directories(NeuralNetworks PRIVATE ../CudaLightKernels)
target_include_directories(NeuralNetworks PRIVATE ../CudaLight)
target_compile_options(NeuralNetworks PRIVATE ${WARNINGS} ${DEBUG_FLAGS})
target_link_libraries(NeuralNetworks CudaLight NeuralNetworkKernels)

add_executable(
        DigitsRecognition
        main.cpp
)
target_include_directories(DigitsRecognition PRIVATE ../CudaLightKernels)
target_include_directories(DigitsRecognition PRIVATE .)
target_compile_options(DigitsRecognition PRIVATE ${WARNINGS} ${DEBUG_FLAGS})
target_link_libraries(DigitsRecognition NeuralNetworks)

find_package(GTest REQUIRED)
add_executable(
        NnUnitTests
        UnitTests/main.cpp
        UnitTests/DataUnitTests.cpp
        UnitTests/NetworkUnitTests.cpp
)
target_include_directories(NnUnitTests PRIVATE NeuralNetworks)
target_include_directories(NnUnitTests PRIVATE ../)
target_include_directories(NnUnitTests PRIVATE ${GTEST_INCLUDE_DIR})
target_compile_options(NnUnitTests PRIVATE ${DEBUG_FLAGS})
target_link_libraries(NnUnitTests gtest pthread NeuralNetworks)