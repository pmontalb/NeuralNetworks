include_guard()

if (LANGUAGES_USE_CUDA)
    set(PATHS_CUDA_PATH "/usr/local")
    mark_as_advanced(PATHS_CUDA_PATH)

    if (NOT LANGUAGES_CUDA_VERSION)
        message(STATUS "Using CUDA 10.2 as nothing specified")
        set(LANGUAGES_CUDA_VERSION 10.2)
    else()
        message(STATUS "Using CUDA${LANGUAGES_CUDA_VERSION}")
    endif()

    set(CMAKE_CUDA_STANDARD ${FLAGS_CPP_STANDARD})
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda-${LANGUAGES_CUDA_VERSION}/bin/nvcc CACHE PATH "" FORCE)

    if (NOT LANGUAGES_CUDA_ARCHITECTURE)
        message(STATUS "Using gen 6.1 as nothing specified")
        set(LANGUAGES_CUDA_ARCHITECTURE 61)
    else()
        message(STATUS "Using gen ${LANGUAGES_CUDA_ARCHITECTURE}")
    endif()

    set(DEFAULT_CUDA_FLAGS -gencode arch=compute_${LANGUAGES_CUDA_ARCHITECTURE},code=sm_${LANGUAGES_CUDA_ARCHITECTURE} -restrict -Xcompiler -Wno-unknown-pragmas --default-stream per-thread)
    if (LANGUAGES_USE_64_BITS)
        set(DEFAULT_CUDA_FLAGS ${DEFAULT_CUDA_FLAGS} -m64)
    endif()
    if (WARNINGS_USE_WALL_AND_WEXTRA)
        set(DEFAULT_CUDA_FLAGS ${DEFAULT_CUDA_FLAGS} --compiler-options -Wall)
    endif()

    if (${CMAKE_BUILD_TYPE} STREQUAL "Release")
        # optimization flags
        set(DEFAULT_CUDA_FLAGS ${DEFAULT_CUDA_FLAGS} -Xptxas -O3)

        if (OPTIMIZATIONS_USE_FAST_MATH)
            # fast math (use intrinsics!)
            set(DEFAULT_CUDA_FLAGS ${DEFAULT_CUDA_FLAGS} --use_fast_math)
        endif()
    endif()

    mark_as_advanced(DEFAULT_CUDA_FLAGS)
endif()