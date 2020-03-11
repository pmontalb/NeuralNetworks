include_guard()

enable_testing()
enable_language(CXX)

if (LANGUAGES_USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    enable_language(CUDA)
endif()