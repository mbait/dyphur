# Backend selection and sanitizer options.
#
# DYPHUR_BACKEND: which AdaptiveCpp compute backend to target.
#   CUDA: NVIDIA GPUs via nvc++/clang+nvptx
#   HIP:  AMD GPUs via ROCm
#   L0:   Intel GPUs via Level Zero
#   OMP:  CPU fallback via OpenMP (also used for ASan/UBSan/TSan builds)
#
# DYPHUR_SANITIZER: sanitizer flavor. OMP backend only — sanitizers + GPU
# is generally not useful.
#   "" (default), ASan, UBSan, TSan

set(DYPHUR_BACKEND "OMP" CACHE STRING "Compute backend: CUDA, HIP, L0, OMP")
set_property(CACHE DYPHUR_BACKEND PROPERTY STRINGS CUDA HIP L0 OMP)

set(DYPHUR_SANITIZER "" CACHE STRING "Sanitizer: '', ASan, UBSan, TSan (OMP only)")
set_property(CACHE DYPHUR_SANITIZER PROPERTY STRINGS "" ASan UBSan TSan)

if(DYPHUR_SANITIZER AND NOT DYPHUR_BACKEND STREQUAL "OMP")
    message(FATAL_ERROR
        "DYPHUR_SANITIZER=${DYPHUR_SANITIZER} requires DYPHUR_BACKEND=OMP (got ${DYPHUR_BACKEND})")
endif()

if(DYPHUR_SANITIZER STREQUAL "ASan")
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
elseif(DYPHUR_SANITIZER STREQUAL "UBSan")
    add_compile_options(-fsanitize=undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=undefined)
elseif(DYPHUR_SANITIZER STREQUAL "TSan")
    add_compile_options(-fsanitize=thread -fno-omit-frame-pointer)
    add_link_options(-fsanitize=thread)
endif()

message(STATUS "dyphur: backend = ${DYPHUR_BACKEND}")
if(DYPHUR_SANITIZER)
    message(STATUS "dyphur: sanitizer = ${DYPHUR_SANITIZER}")
endif()

# Map DYPHUR_BACKEND to ACPP_TARGETS (AdaptiveCpp compilation target string).
# ACPP_TARGETS can be overridden on the command line, e.g.:
#   -DACPP_TARGETS="cuda:sm_90"   for H100
#   -DACPP_TARGETS="hip:gfx1100"  for RX 7900
if(NOT DEFINED ACPP_TARGETS)
    if(DYPHUR_BACKEND STREQUAL "CUDA")
        # Default: RTX 3060 (Ampere sm_86). Override for other GPUs.
        set(ACPP_TARGETS "cuda:sm_86"
            CACHE STRING "AdaptiveCpp compilation targets")
    elseif(DYPHUR_BACKEND STREQUAL "HIP")
        set(ACPP_TARGETS "hip"
            CACHE STRING "AdaptiveCpp compilation targets")
    elseif(DYPHUR_BACKEND STREQUAL "L0")
        set(ACPP_TARGETS "level_zero:gpu"
            CACHE STRING "AdaptiveCpp compilation targets")
    else()
        set(ACPP_TARGETS "omp"
            CACHE STRING "AdaptiveCpp compilation targets")
    endif()
endif()

find_package(AdaptiveCpp REQUIRED)
message(STATUS "dyphur: ACPP_TARGETS = ${ACPP_TARGETS}")
