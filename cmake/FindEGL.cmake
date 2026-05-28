# FindEGL.cmake — creates EGL::EGL imported target using system libEGL.
# Required because CMake 3.28 ships no built-in FindEGL, but Magnum's
# FindMagnum.cmake calls find_package(EGL) to get EGL::EGL.

find_path(EGL_INCLUDE_DIR NAMES EGL/egl.h
    PATHS /usr/include /usr/local/include)

find_library(EGL_LIBRARY NAMES EGL
    PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(EGL DEFAULT_MSG EGL_LIBRARY EGL_INCLUDE_DIR)

if(EGL_FOUND AND NOT TARGET EGL::EGL)
    add_library(EGL::EGL UNKNOWN IMPORTED)
    set_target_properties(EGL::EGL PROPERTIES
        IMPORTED_LOCATION "${EGL_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${EGL_INCLUDE_DIR}")
endif()

mark_as_advanced(EGL_INCLUDE_DIR EGL_LIBRARY)
