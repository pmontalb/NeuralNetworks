include_guard()

set(WARNING_FLAGS "")
mark_as_advanced(WARNING_FLAGS)

if (WARNINGS_USE_WALL_AND_WEXTRA)
    message(STATUS "Use -Wall and -Wextra")
    set(WARNING_FLAGS ${WARNING_FLAGS} -Wall -Wextra)
endif()
if (WARNINGS_TREAT_WARNINGS_AS_ERROR)
    message(STATUS "Use -Werror")
    set(WARNING_FLAGS ${WARNING_FLAGS} -Werror)
endif()

# disable basic warnings
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(WARNING_FLAGS ${WARNING_FLAGS} -Wno-unused-function)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(WARNING_FLAGS ${WARNING_FLAGS} -Wno-unknown-pragmas -Wno-unused-function -Wno-unused-macros)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    message(FATAL "icc not supported")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    message(FATAL "MSVC not supported")
endif()

set(PEDANTIC_WARNING_FLAGS "")
mark_as_advanced(PEDANTIC_WARNING_FLAGS)

if (WARNINGS_USE_PEDANTIC_WARNINGS)
    message(STATUS "Use pedantic warnings")

    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        set(PEDANTIC_WARNING_FLAGS ${WARNING_FLAGS} -Weverything
                -Wno-unknown-cuda-version
                -Wno-c++98-compat
                -Wno-c++98-compat-pedantic
                -Wno-reserved-id-macro
                -Wno-switch-enum
                -Wno-unknown-pragmas
                -Wno-disabled-macro-expansion
                -Wno-unneeded-internal-declaration
                -Wno-weak-vtables
                -Wno-padded
                -Wno-unused-macros
                -Wno-exit-time-destructors)
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        # https://raw.githubusercontent.com/Barro/compiler-warnings/master/gcc/warnings-gcc-top-level-8.txt
        set(PEDANTIC_WARNING_FLAGS ${WARNING_FLAGS}
                -Wabi-tag
                -Waggregate-return
                -Waggressive-loop-optimizations
                -Wall
                -Walloc-zero
                -Walloca
                -Wattribute-alias
                -Wattributes
                -Wbuiltin-declaration-mismatch
                -Wbuiltin-macro-redefined
                -Wcast-align
                -Wcast-align=strict
                -Wcast-qual
                -Wconditionally-supported
                -Wconversion
                -Wconversion-null
                -Wcoverage-mismatch
                -Wcpp
                -Wctor-dtor-privacy
                -Wdate-time
                -Wdelete-incomplete
                -Wdeprecated
                -Wdeprecated-declarations
                -Wdisabled-optimization
                -Wdiv-by-zero
                -Wdouble-promotion
                -Wduplicated-branches
                -Wduplicated-cond
                -Weffc++
                -Wextra-semi
                -Wfloat-equal
                -Wformat-signedness
                -Wfree-nonheap-object
                -Whsa
                -Wif-not-aligned
                -Wignored-attributes
                -Wimport # DUMMY switch
                -Winherited-variadic-ctor
                -Winline
                -Wint-to-pointer-cast
                -Winvalid-memory-model
                -Winvalid-offsetof
                -Winvalid-pch
                -Wliteral-suffix
                -Wlogical-op
                -Wlto-type-mismatch
                -Wmissing-declarations
                -Wmissing-include-dirs
                -Wmultichar
                -Wmultiple-inheritance
                -Wnoexcept
                -Wnon-template-friend
                -Wnull-dereference
                -Wodr
                -Wold-style-cast
                -Woverflow
                -Woverloaded-virtual
                -Wpacked
                -Wpacked-bitfield-compat
                #-Wpadded
                -Wpedantic
                -Wpmf-conversions
                -Wpointer-compare
                -Wpragmas
                -Wredundant-decls
                -Wregister
                -Wreturn-local-addr
                -Wscalar-storage-order
                -Wshadow
                -Wshift-count-negative
                -Wshift-count-overflow
                -Wshift-negative-value
                -Wsign-promo
                -Wsizeof-array-argument
                -Wstack-protector
                -Wstrict-aliasing
                -Wstrict-null-sentinel
                -Wstrict-overflow
                -Wsubobject-linkage
                -Wsuggest-attribute=cold
                -Wsuggest-attribute=const
                -Wsuggest-attribute=format
                -Wsuggest-attribute=malloc
                -Wsuggest-attribute=noreturn
                -Wsuggest-attribute=pure
                -Wsuggest-final-methods
                -Wsuggest-final-types
                -Wsuggest-override
                -Wswitch-bool
                -Wswitch-default
                #-Wswitch-enum
                -Wswitch-unreachable
                -Wsync-nand
                -Wsynth
                -Wterminate
                -Wtrampolines
                #-Wundef
                -Wunreachable-code # DUMMY switch
                -Wunsafe-loop-optimizations # DUMMY switch
                -Wunused-result
                -Wuseless-cast
                -Wvarargs
                -Wvector-operation-performance
                -Wvirtual-inheritance
                -Wvirtual-move-assign
                -Wvla
                -Wwrite-strings
                -Wzero-as-null-pointer-constant)

    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        message(FATAL "icc not supported")
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
        message(FATAL "MSVC not supported")
    endif()

endif()