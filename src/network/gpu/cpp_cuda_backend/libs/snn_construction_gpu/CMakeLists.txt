cmake_minimum_required(VERSION 3.6)

project(snn_construction_gpu)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)
find_package(CUDA REQUIRED)
# find_library(${CUDA_cusparse_LIBRARY} PATHS "/usr/local/cuda/lib64" NO_DEFAULT_PATH) 

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 61)
endif()


find_package(PythonInterp 3.9 REQUIRED)
find_package(PythonLibs 3.9 REQUIRED)
find_package(OpenGL REQUIRED)

get_filename_component(PROJECT_INC_DIRECTORY ../../../../../../inc ABSOLUTE)

include_directories(
    ./src/
    /usr/local/cuda/lib64/
    ${PYTHON_INCLUDE_DIRS}
    ${PROJECT_INC_DIRECTORY}
    ${OPENGL_INCLUDE_DIR}
)

link_directories(
    /usr/local/cuda/lib64
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5/libs/x64"
)


if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -Xcompiler -fPIC)

set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

cuda_add_library(
    snn_simulation_gpu SHARED
    ./src/simulation/snn_simulation_bindings.cu
    ./src/simulation/snn_simulation.cu
    ./src/utils/launch_parameters.cu
    ./src/utils/curand_states.cu
)

cuda_add_library(
    snn_construction_gpu SHARED
    ./src/construction/snn_construction_bindings.cu
    ./src/construction/snn_representation.cu
    ./src/utils/launch_parameters.cu
    ./src/utils/curand_states.cu
)

target_link_libraries(
    snn_simulation_gpu
    ${CUDA_cusparse_LIBRARY}
)

target_link_libraries(
    snn_construction_gpu
    ${CUDA_cusparse_LIBRARY}
)

set_target_properties(snn_simulation_gpu PROPERTIES PREFIX "")
set_target_properties(snn_construction_gpu PROPERTIES PREFIX "")

IF (WIN32)
    set_target_properties(snn_simulation_gpu PROPERTIES SUFFIX ".pyd")
    set_target_properties(snn_construction_gpu PROPERTIES SUFFIX ".pyd")
ENDIF()