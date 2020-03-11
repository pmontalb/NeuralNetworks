include_guard()

cmake_minimum_required(VERSION 3.14)
project(NeuralNetworks)

include(cmake/All.cmake)
set(LANGUAGES_USE_CUDA ON CACHE BOOL "" FORCE)

# CudaLight
add_subdirectory(../CudaLight ${CMAKE_BINARY_DIR}/CudaLight EXCLUDE_FROM_ALL)

create_cuda_library(
    NAME
        NeuralNetworkKernels
    SOURCES
        NeuralNetworkKernels/ObjectiveFunctions.cu
    PUBLIC_INCLUDE_DIRECTORIES
        NeuralNetworkKernels
    DEPENDENCIES
        CudaLightKernels
)

create_library(
    NAME
        NeuralNetworks
    SOURCES
        NeuralNetworks/NeuralNetworksManager.cpp
    DEPENDENCIES
        CudaLight NeuralNetworkKernels
    PUBLIC_INCLUDE_DIRECTORIES
        NeuralNetworks .
)

create_executable(
    NAME
        DigitsRecognition
    SOURCES
        main.cpp
    DEPENDENCIES
        NeuralNetworks
)

find_package(GTest REQUIRED)
create_test(
    NAME
        NnUnitTests
    SOURCES
        UnitTests/main.cpp
        UnitTests/DataUnitTests.cpp
        UnitTests/NetworkUnitTests.cpp
    DO_NOT_USE_WARNINGS
    DO_NOT_USE_PEDANTIC_WARNINGS
    PUBLIC_INCLUDE_DIRECTORIES
        ${GTEST_INCLUDE_DIR}
    DEPENDENCIES
        NeuralNetworks
    SYSTEM_DEPENDENCIES
        gtest pthread
)