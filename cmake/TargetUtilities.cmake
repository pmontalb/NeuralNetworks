include_guard()

function(create_executable)
    create_target(TARGET_TYPE "EXE" ${ARGN})
endfunction()

function(create_library)
    create_target(TARGET_TYPE "LIB" ${ARGN})
endfunction()

function(create_static_library)
    create_target(TARGET_TYPE "STATIC_LIB" ${ARGN})
endfunction()

function(create_test)
    create_target(TARGET_TYPE "TEST" ${ARGN})
endfunction()

function(create_cuda_executable)
    create_cuda_target(TARGET_TYPE "EXE" ${ARGN})
endfunction()

function(create_cuda_library)
    create_cuda_target(TARGET_TYPE "LIB" ${ARGN})
endfunction()

function(create_cuda_static_library)
    create_cuda_target(TARGET_TYPE "STATIC_LIB" ${ARGN})
endfunction()

function(create_cuda_test)
    create_cuda_target(TARGET_TYPE "TEST" ${ARGN})
endfunction()

function(create_target)
    cmake_parse_arguments(
            PREFIX
            "DO_NOT_USE_WARNINGS;DO_NOT_USE_PEDANTIC_WARNINGS"
            "NAME;TARGET_TYPE"
            "SOURCES;PUBLIC_INCLUDE_DIRECTORIES;PRIVATE_INCLUDE_DIRECTORIES;DEPENDENCIES;SYSTEM_DEPENDENCIES"
            ${ARGN}
    )

    if (PREFIX_TARGET_TYPE STREQUAL "EXE")
        add_executable(${PREFIX_NAME} ${PREFIX_SOURCES} ${DEPENDENCIES})
    elseif (PREFIX_TARGET_TYPE STREQUAL "LIB")
        add_library(${PREFIX_NAME} ${PREFIX_SOURCES} ${DEPENDENCIES})
    elseif (PREFIX_TARGET_TYPE STREQUAL "STATIC_LIB")
        add_library(${PREFIX_NAME} STATIC ${PREFIX_SOURCES} ${DEPENDENCIES})
    elseif (PREFIX_TARGET_TYPE STREQUAL "TEST")
        add_executable(${PREFIX_NAME} ${PREFIX_SOURCES} ${DEPENDENCIES})
        add_test(NAME ${PREFIX_NAME})
    endif()

    if (NOT PREFIX_DO_NOT_USE_PEDANTIC_WARNINGS AND PREFIX_DO_NOT_USE_WARNINGS)
        target_compile_options(${PREFIX_NAME} PRIVATE ${PEDANTIC_WARNING_FLAGS})
    elseif (NOT PREFIX_DO_NOT_USE_PEDANTIC_WARNINGS)
        target_compile_options(${PREFIX_NAME} PRIVATE ${WARNING_FLAGS})
    endif()

    target_include_directories(${PREFIX_NAME} PUBLIC ${PREFIX_PUBLIC_INCLUDE_DIRECTORIES})
    target_include_directories(${PREFIX_NAME} PRIVATE ${PREFIX_PRIVATE_INCLUDE_DIRECTORIES})
    target_compile_definitions(${PREFIX_NAME} PRIVATE ${STL_DEBUG_FLAGS})
    target_link_libraries(${PREFIX_NAME} ${DEPENDENCIES})
endfunction()

function(create_cuda_target)
    cmake_parse_arguments(
            PREFIX
            "DO_NOT_USE_WARNINGS;DO_NOT_USE_PEDANTIC_WARNINGS"
            "NAME;TARGET_TYPE"
            "SOURCES;PUBLIC_INCLUDE_DIRECTORIES;PRIVATE_INCLUDE_DIRECTORIES;DEPENDENCIES;SYSTEM_DEPENDENCIES"
            ${ARGN}
    )

    if (PREFIX_TARGET_TYPE STREQUAL "EXE")
        add_executable(${PREFIX_NAME} ${PREFIX_SOURCES} ${DEPENDENCIES})
    elseif (PREFIX_TARGET_TYPE STREQUAL "LIB")
        add_library(${PREFIX_NAME} ${PREFIX_SOURCES} ${DEPENDENCIES})
    elseif (PREFIX_TARGET_TYPE STREQUAL "STATIC_LIB")
        add_library(${PREFIX_NAME} STATIC ${PREFIX_SOURCES} ${DEPENDENCIES})
    elseif (PREFIX_TARGET_TYPE STREQUAL "TEST")
        add_executable(${PREFIX_NAME} ${PREFIX_SOURCES} ${DEPENDENCIES})
        add_test(NAME ${PREFIX_NAME})
    endif()

    target_include_directories(${PREFIX_NAME} PUBLIC ${PREFIX_PUBLIC_INCLUDE_DIRECTORIES})
    target_include_directories(${PREFIX_NAME} PRIVATE ${PREFIX_PRIVATE_INCLUDE_DIRECTORIES})
    target_link_libraries(${PREFIX_NAME} ${DEPENDENCIES})
    target_link_libraries(${PREFIX_NAME} -lcudart -lcublas -lcusolver -lcusparse)
    target_compile_options(${PREFIX_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: ${DEFAULT_CUDA_FLAGS}>)
endfunction()