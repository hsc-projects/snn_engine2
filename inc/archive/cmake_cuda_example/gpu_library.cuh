#include <sstream>
#include <iostream>

#include <windows.h>


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

template <typename T>
__global__ void kernel
(T *vec, T scalar, int num_elements)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        vec[idx] = vec[idx] * scalar;
    }
}

template <typename T>
void run_kernel
(T *vec, T scalar, int num_elements)
{
    dim3 dimBlock(256, 1, 1);
    dim3 dimGrid(ceil((T)num_elements / dimBlock.x));
    
    kernel<T><<<dimGrid, dimBlock>>>
        (vec, scalar, num_elements);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::stringstream strstr;
        strstr << "run_kernel launch failed" << std::endl;
        strstr << "dimBlock: " << dimBlock.x << ", " << dimBlock.y << std::endl;
        strstr << "dimGrid: " << dimGrid.x << ", " << dimGrid.y << std::endl;
        strstr << cudaGetErrorString(error);
        throw strstr.str();
    }
}

template <typename T>
void map_array(pybind11::array_t<T> vec, T scalar)
{
    pybind11::buffer_info ha = vec.request();

    if (ha.ndim != 1) {
        std::stringstream strstr;
        strstr << "ha.ndim != 1" << std::endl;
        strstr << "ha.ndim: " << ha.ndim << std::endl;
        throw std::runtime_error(strstr.str());
    }

    int size = ha.shape[0];
    int size_bytes = size*sizeof(T);
    T *gpu_ptr;
    cudaError_t error = cudaMalloc(&gpu_ptr, size_bytes);

    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }

    T* ptr = reinterpret_cast<T*>(ha.ptr);
    error = cudaMemcpy(gpu_ptr, ptr, size_bytes, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }

    run_kernel<T>(gpu_ptr, scalar, size);

    error = cudaMemcpy(ptr, gpu_ptr, size_bytes, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }

    error = cudaFree(gpu_ptr);
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

typedef unsigned int uint;

struct Pet {

    bool bmapped = false;

    uint id{};
	size_t size{};

    Pet(const std::string &name) : name(name) { }
    void setName(const std::string &name_) { name = name_; }
    const std::string &getName() const { return name; }

    cudaGraphicsResource* buffer_pt{ nullptr };
    
    void register_buffer(uint b);

    std::string name;
};



