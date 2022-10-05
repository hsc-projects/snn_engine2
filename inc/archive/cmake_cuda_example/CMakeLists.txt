cmake_minimum_required(VERSION 3.6)

project(gpu_library)

find_package(CUDA REQUIRED)
find_package(PythonInterp 3.6 REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)
find_package(OpenGL REQUIRED)


get_filename_component(PROJECT_INC_DIRECTORY ../../../inc ABSOLUTE)

include_directories(
    .
    ${PYTHON_INCLUDE_DIRS}
    ${PROJECT_INC_DIRECTORY}/pybind11/include
    ${OPENGL_INCLUDE_DIR}
    
)

link_directories(
    /usr/local/cuda/lib64
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5/lib/x64"
)


if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()


cuda_add_library(
    gpu_library SHARED bindings.cu pet2.cpp gpu_library.cu
)

target_link_libraries(
    gpu_library
    ${PYTHON_LIBRARIES}
    cudart
)

set_target_properties(gpu_library PROPERTIES PREFIX "")

IF (WIN32)
    set_target_properties(gpu_library PROPERTIES SUFFIX ".pyd")
ENDIF()