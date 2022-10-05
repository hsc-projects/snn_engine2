#include <cuda_opengl_interop.cuh>

void Pet::register_buffer(uint b){
    id = b;
    cudaGraphicsGLRegisterBuffer(&buffer_pt, id, cudaGraphicsRegisterFlagsNone);
    bmapped = true;
}