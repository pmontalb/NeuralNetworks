include_guard()

include(cmake/Options.cmake)
include(cmake/Configurations.cmake)
include(cmake/Warnings.cmake)
include(cmake/OptimizationFlags.cmake)
include(cmake/DebugFlags.cmake)
include(cmake/SanitizerFlags.cmake)
include(cmake/CodeCoverageFlags.cmake)

if (NOT FLAGS_CPP_STANDARD)
    message(STATUS "Using C++14 standard as nothing specified")
    set(FLAGS_CPP_STANDARD 14)
else()
    message(STATUS "Using C++${FLAGS_CPP_STANDARD} standard")
endif()
set(CMAKE_CXX_STANDARD ${FLAGS_CPP_STANDARD})

set(STL_DEBUG_FLAGS "")
if (${CMAKE_BUILD_TYPE} STREQUAL "Debug" AND DEBUG_USE_STL_DEBUG)
    message(STATUS "Use STL debug")
    set(STL_DEBUG_FLAGS -D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC)
endif()

