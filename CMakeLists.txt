cmake_minimum_required(VERSION 3.18)
project(vibeflow LANGUAGES CXX CUDA)

# ----------------------------------
# CUDA standard
# ----------------------------------
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Optional:
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------------------------
# Find all .cu files recursively
# ----------------------------------
file(GLOB_RECURSE CUDA_SOURCES
    "${CMAKE_SOURCE_DIR}/src/*.*"
)

# ----------------------------------
# Exclude test.cu
# ----------------------------------
list(FILTER CUDA_SOURCES EXCLUDE REGEX ".*test\\.cu$")

# ----------------------------------
# Create executable
# ----------------------------------
add_executable(${PROJECT_NAME}
    ${CUDA_SOURCES}
)

# ----------------------------------
# Include directories
# ----------------------------------
target_include_directories(${PROJECT_NAME} PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    # "${CMAKE_SOURCE_DIR}/src/VibeflowFunctions"
)

# ----------------------------------
# Optional CUDA architecture
# (adjust for your GPU)
# Example: 86 = RTX 30xx
# ----------------------------------
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_ARCHITECTURES 75
)

# ----------------------------------
# Optional optimization flags
# ----------------------------------
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>
)
