include_guard()

option(BUILD_SHARED_LIBS OFF)

option(WARNINGS_TREAT_WARNINGS_AS_ERROR "Use -Werror" ON)
option(WARNINGS_USE_WALL_AND_WEXTRA "Use -Wall and -Wextra" ON)
option(WARNINGS_USE_PEDANTIC_WARNINGS "Use -Weverything" ON)

option(DEBUG_USE_STL_DEBUG "Use GLIBCXX_DEBUG and GLIBCXX_DEBUG_PEDANTIC" ON)

option(LANGUAGES_USE_64_BITS "Use 64 bits" ON)
mark_as_advanced(LANGUAGES_USE_64_BITS)

option(LANGUAGES_USE_CUDA "Depends on CUDA" OFF)

option(OPTIMIZATIONS_GENERATE_DEBUG_INFO "Use -g in release mode" ON)
option(OPTIMIZATIONS_USE_OFAST "Use -Ofast in release mode instead of O3" OFF)
option(OPTIMIZATIONS_USE_MEMORY_OPTIMIZATION "Use Graphite/Polly in release mode" OFF)
option(OPTIMIZATIONS_USE_FAST_MATH "Use -ffast-math in release mode" OFF)
option(OPTIMIZATIONS_USE_LTO "Use Link-Time Optimization in release mode" OFF)