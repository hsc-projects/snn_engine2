import ctypes
import numba.cuda
import numpy as np
import pandas as pd

from pycuda.compiler import SourceModule
import pycuda.driver
import pycuda.gpuarray
import pycuda.gpuarray
# noinspection PyUnresolvedReferences
from pycuda.gl import (
    RegisteredBuffer,
    RegisteredImage,
    RegisteredMapping,
    graphics_map_flags
)
from pycuda.tools import dtype_to_ctype
import torch
from typing import Optional, Union
from vispy.gloo import gl


def shape_size(shape, unit=None):
    if not unit:
        return np.array(shape).cumprod()[-1]
    elif isinstance(unit, int):
        return shape_size(shape) / unit
    elif unit == 'mb':
        return shape_size(shape, pow(10, 6))


def reshape_wrt_size(shape, max_size_mb, n_bytes=4):

    if ((len(shape) != 2)
            or (not isinstance(shape[0], int))
            or (not isinstance(shape[1], int))):
        raise TypeError

    max_size = np.ceil((max_size_mb * pow(10, 6)) / n_bytes)
    shape_size_ = shape_size(shape)

    if shape_size_ > max_size:
        max_square_length = np.sqrt(max_size)
        new_shape = np.array([max_square_length, max_square_length])

        for i in range(len(shape)):
            if shape[i] < new_shape[i]:
                new_shape[i] = shape[i]
                if i < (len(shape) - 1):
                    new_shape[i + 1] += (new_shape[i + 1] - new_shape[i])

        sqrt_new_shape = tuple(np.sqrt(new_shape))

        for i in range(len(shape)):
            while ((((shape[i] % (new_shape[i] - sqrt_new_shape[i])) - (shape[i] % new_shape[i])) > 0)
                   and ((shape[i] % new_shape[i]) > 0)):
                """
                Goal: find a better x such that batch sizes become closer to each other.
                As long a reducing x by sqrt(x) does not result in int(old_x/x) increasing, reduce x by sqrt(x). 
                """
                new_shape[i] -= sqrt_new_shape[i]

        new_shape_size = np.array(new_shape).cumprod()[-1]

        if not new_shape_size < max_size:
            raise ValueError

        return tuple(new_shape)
    return shape


class ExternalMemory(object):
    """
    Provide an externally managed memory.
    Interface requirement: __cuda_memory__, device_ctypes_pointer, _cuda_memsize_
    """
    __cuda_memory__ = True

    def __init__(self, ptr, size):
        self.device_ctypes_pointer = ctypes.c_void_p(ptr)
        self._cuda_memsize_ = size


class GPUArrayConfig:

    def __init__(self, shape=None, strides=None, dtype=None, stream=0, device: torch.device = None):

        if device is None:
            raise ValueError('device is None.')
        elif isinstance(device, int):
            device = torch.device(device)

        self.shape: Optional[tuple] = shape
        self.strides:  tuple = strides
        self.dtype: np.dtype = dtype

        self.stream: int = stream

        self.device: torch.device = device

    @classmethod
    def from_cpu_array(cls, cpu_array, dev: torch.device = None, stream=0):
        shape: tuple = cpu_array.shape
        strides:  tuple = cpu_array.strides
        dtype: np.dtype = cpu_array.dtype
        return GPUArrayConfig(shape=shape, strides=strides, dtype=dtype, stream=stream, device=dev)


class RegisteredGPUArray:

    def __init__(self,
                 gpu_data: Union[ExternalMemory, pycuda.driver.Array] = None,
                 reg: RegisteredBuffer = None,
                 mapping: RegisteredMapping = None,
                 ptr: int = None,
                 config: GPUArrayConfig = None,
                 id_: Optional[int] = None):

        numba.cuda.select_device(config.device.index)

        self.reg: RegisteredBuffer = reg
        self.mapping: RegisteredMapping = mapping
        self.ptr: int = ptr
        self.conf: GPUArrayConfig = config

        self.gpu_data: Union[ExternalMemory, pycuda.driver.Array] = gpu_data

        self._id = id_

        self.tensor: Optional[torch.Tensor] = None
        # noinspection PyUnresolvedReferences
        self._numba_device_array: Optional[numba.cuda.cudadrv.devicearray.DeviceNDArray] = None

    @property
    def ctype_ptr(self):
        return self.gpu_data.device_ctypes_pointer

    @property
    def size(self):
        # noinspection PyProtectedMember
        return self.gpu_data._cuda_memsize_

    # noinspection PyArgumentList
    @classmethod
    def _read_buffer(cls, buffer_id):

        reg = RegisteredBuffer(buffer_id)
        mapping: RegisteredMapping = reg.map(None)
        ptr, size = mapping.device_ptr_and_size()
        gpu_data = ExternalMemory(ptr, size)
        mapping.unmap()

        return dict(gpu_data=gpu_data, reg=reg, mapping=mapping, ptr=ptr, id_=buffer_id)

    # noinspection PyArgumentList
    @classmethod
    def _read_texture(cls, texture_id):

        # noinspection PyUnresolvedReferences
        reg = RegisteredImage(texture_id, gl.GL_TEXTURE_3D,
                              graphics_map_flags.NONE
                              # graphics_map_flags.WRITE_DISCARD
                              )
        mapping: RegisteredMapping = reg.map(None)
        gpu_data = mapping.array(0, 0)
        ptr = gpu_data.handle
        mapping.unmap()

        return dict(gpu_data=gpu_data, reg=reg, mapping=mapping, ptr=ptr, id_=texture_id)

    @classmethod
    def from_buffer(cls, buffer, config: GPUArrayConfig = None, cpu_array: np.array = None):

        if config is not None:
            assert cpu_array is None
        else:
            config = GPUArrayConfig.from_cpu_array(cpu_array)
        return RegisteredGPUArray(config=config, **cls._read_buffer(buffer))

    def map(self):
        self.reg.map(None)

    def unmap(self):
        # noinspection PyArgumentList
        self.mapping.unmap()

    # @property
    # def tensor(self) -> torch.Tensor:
    #     self.map()
    #     if self._tensor is None:
    #         self._tensor = torch.as_tensor(self.device_array, device=self.conf.device)
    #     return self._tensor

    def data_ptr(self) -> int:
        return self.tensor.data_ptr()

    @property
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.tensor.cpu().numpy())

    def unregister(self):
        # noinspection PyArgumentList
        self.reg.unregister()

    def copy_to_host(self):
        return self._numba_device_array.copy_to_host()


class RegisteredVBO(RegisteredGPUArray):

    def __init__(self, buffer, shape, device):
        stride = shape[1]
        nbytes_float32 = 4

        config = GPUArrayConfig(shape=shape, strides=(stride * nbytes_float32, nbytes_float32),
                                dtype=np.float32, device=device)
        RegisteredGPUArray.__init__(self, config=config, **self._read_buffer(buffer))

        # noinspection PyUnresolvedReferences
        self._numba_device_array = numba.cuda.cudadrv.devicearray.DeviceNDArray(
            shape=self.conf.shape,
            strides=self.conf.strides,
            dtype=self.conf.dtype,
            stream=self.conf.stream,
            gpu_data=self.gpu_data)

        self.tensor = torch.as_tensor(self._numba_device_array, device=self.conf.device)


class RegisteredIBO(RegisteredGPUArray):

    def __init__(self, buffer, shape, device):
        stride = shape[1]
        nbytes_int32 = 4

        config = GPUArrayConfig(shape=shape, strides=(stride * nbytes_int32, nbytes_int32),
                                dtype=np.int32, device=device)
        RegisteredGPUArray.__init__(self, config=config, **self._read_buffer(buffer))

        # noinspection PyUnresolvedReferences
        self._numba_device_array = numba.cuda.cudadrv.devicearray.DeviceNDArray(
            shape=self.conf.shape,
            strides=self.conf.strides,
            dtype=self.conf.dtype,
            stream=self.conf.stream,
            gpu_data=self.gpu_data)

        self.tensor = torch.as_tensor(self._numba_device_array, device=self.conf.device)


class RegisteredTexture3D(RegisteredGPUArray):

    def __init__(self, buffer, shape, device, cpu_data):

        stride1 = shape[2]
        stride2 = shape[1]
        nbytes_int16 = 4

        config = GPUArrayConfig(shape=shape, strides=(stride2 * (nbytes_int16 ** 2),
                                                      stride1 * nbytes_int16,
                                                      nbytes_int16),
                                dtype=np.float32, device=device)

        RegisteredGPUArray.__init__(self, config=config, **self._read_texture(buffer))

        self.tensor: torch.Tensor = torch.zeros(self.conf.shape, device=self.conf.device, dtype=torch.float32)

        self._cpy_tnsr2tex = pycuda.driver.Memcpy3D()
        self._cpy_tnsr2tex.set_src_device(self.tensor.data_ptr())
        self._cpy_tnsr2tex.set_dst_array(self.gpu_data)

        self._cpy_tex2tnsr = pycuda.driver.Memcpy3D()
        self._cpy_tex2tnsr.set_src_array(self.gpu_data)
        self._cpy_tex2tnsr.set_dst_device(self.tensor.data_ptr())

        self._cpy_tnsr2tex.width_in_bytes = self._cpy_tex2tnsr.width_in_bytes = nbytes_int16 * self.conf.shape[2]

        self._cpy_tnsr2tex.src_pitch = self._cpy_tex2tnsr.src_pitch = nbytes_int16 * self.conf.shape[2]

        self._cpy_tnsr2tex.src_height = self._cpy_tnsr2tex.height\
            = self._cpy_tex2tnsr.src_height = self._cpy_tex2tnsr.height = self.conf.shape[1]

        self._cpy_tnsr2tex.depth = self._cpy_tex2tnsr.depth = self.conf.shape[0]

        # self.cpy_tnsr2tex()
        self.cpy_tex2tnsr(cpu_data)

    def cpy_tnsr2tex(self, cpu_data=None):
        """
        TODO: Restrict copying to actually modified data.
        """
        self.map()

        # print(self.tensor[0, 0])

        if cpu_data is not None:
            self.tensor[:] = torch.from_numpy(cpu_data)
        torch.cuda.synchronize()
        self._cpy_tnsr2tex()
        # self._cpy_tex2tnsr()
        #
        # print(self.tensor[0, 0, 0])
        # t = self.tensor.cpu().numpy()
        # assert (((cpu_data - t) == 0).all())
        # print((t == cpu_data).all())
        return

    def cpy_tex2tnsr(self, cpu_data=None):
        self.map()
        torch.cuda.synchronize()
        self._cpy_tex2tnsr()
        if cpu_data is not None:
            # print(cpu_data)
            t = self.tensor.cpu().numpy()
            # print(t)
            assert (((cpu_data - t) == 0).all())
            # print((t == cpu_data).all())
        return


def copy_texture(shape, arr_cpu):
    """ Unsuccessful attempts to modify the volume data in-place (instead of copying from a torch.Tensor). """

    precision = np.float32
    # arr_cpu = np.zeros((shape[0], shape[1], shape[2]), order="C", dtype=np.float32)
    # arr_cpu[:] = np.random.rand(shape[0], shape[1], shape[2])[:]
    arr_gpu = pycuda.gpuarray.to_gpu(arr_cpu)

    kernel_read_write_surface = """
    #include <pycuda-helpers.hpp>

    surface<void, cudaSurfaceType3D> mtx_tex;

    __global__ void copy_texture(cuPres *dest, int rw)
    {
      //int row   = blockIdx.x*blockDim.x + threadIdx.x;
      //int col   = blockIdx.y*blockDim.y + threadIdx.y;
      //int slice = blockIdx.z*blockDim.z + threadIdx.z;
      //int tid = row + col*blockDim.x*gridDim.x + slice*blockDim.x*gridDim.x*blockDim.y*gridDim.y;

      int row   = blockIdx.x*blockDim.x + threadIdx.x;
      int col   = blockIdx.y*blockDim.y + threadIdx.y;
      int slice = blockIdx.z*blockDim.z + threadIdx.z;
      int tid = row + blockDim.x * gridDim.x * (col + slice * blockDim.y*gridDim.y);

      if (rw==0){
         cuPres aux = dest[tid];
         surf3Dwrite(aux, mtx_tex, slice * sizeof(float), col, row, cudaBoundaryModeClamp);}
      else {
         cuPres aux = 0;
         surf3Dread(&aux, mtx_tex,  slice* sizeof(float), col, row, cudaBoundaryModeClamp);
      dest[tid] = aux;
      }
    }
    """
    kernel_read_write_surface = kernel_read_write_surface.replace("fpName", dtype_to_ctype(precision))
    kernel_read_write_surface = kernel_read_write_surface.replace("cuPres", dtype_to_ctype(precision))
    module_read_write_surface = SourceModule(kernel_read_write_surface)

    copy_texture = module_read_write_surface.get_function("copy_texture")
    mtx_tex = module_read_write_surface.get_surfref("mtx_tex")

    cuda_block = (8, 8, 8)
    if cuda_block[0] > max(shape):
        cuda_block = (shape[0], shape[1], shape[2])
    cuda_grid = (
        shape[0] // cuda_block[0] + 1 * (shape[0] % cuda_block[0] != 0),
        shape[1] // cuda_block[1] + 1 * (shape[1] % cuda_block[1] != 0),
        shape[2] // cuda_block[2] + 1 * (shape[2] % cuda_block[2] != 0),
    )
    copy_texture.prepare("Pi")

    arr_gpu2 = pycuda.gpuarray.zeros_like(arr_gpu)
    cuda_arr2 = pycuda.driver.gpuarray_to_array(arr_gpu2, "C", allowSurfaceBind=True)

    arr_cpu2 = arr_gpu.get()  # To remember original array
    mtx_tex.set_array(cuda_arr2)
    # mtx_tex.set_array(self.gpu_data)
    copy_texture.prepared_call(cuda_grid, cuda_block, arr_gpu.gpudata, np.int32(0))
    copy_texture.prepared_call(cuda_grid, cuda_block, arr_gpu2.gpudata, np.int32(1))

    arr_cpu3 = arr_gpu2.get()

    assert np.sum(np.abs(arr_cpu3 - arr_cpu2)) == np.array(0, dtype=np.float32)
    # print(arr_cpu)
    # print(arr_cpu2)
    # print(arr_cpu3)
    print()
    # mod = SourceModule(
    # """
    #
    # texture<float, 3, cudaReadModeElementType> mtx_tex;
    # // texture<float, 3, cudaReadModeElementType> mtx_tex2;
    # surface<void,  3> mtx_srf;
    #
    # //surface<void, cudaSurfaceType3D> mtx_srf;
    #
    # __global__ void copy_texture(float *dest, float *og)
    # {
    #   int x = threadIdx.x;
    #   int y = threadIdx.y;
    #   int z = threadIdx.z;
    #   int dx = blockDim.x;
    #   int dy = blockDim.y;
    #   int i = (z*dy + y)*dx + x;
    #   dest[i] = tex3D(mtx_tex, x, y, z);
    #
    #
    #   // surface writes need byte offsets for x!
    #   surf3Dwrite(5.f, mtx_srf, z * sizeof(float), y, x, cudaBoundaryModeTrap);
    #
    #   float data = 1.;
    #   surf3Dread(&data, mtx_srf, z* sizeof(float), y, x, cudaBoundaryModeTrap);
    #
    #   printf("(%d, %d, %d)[%d]: %f, %f, %f \\n", x, y, z, i, dest[i], data, og[i]);
    #
    #   // printf("%d %f ", i, dest[i]);
    #   //dest[i] = data;
    # }
    # """
    # )
    #
    # from pycuda.tools import dtype_to_ctype
    #
    # myKernRW = """
    #  #include <pycuda-helpers.hpp>
    #
    #  surface<void, cudaSurfaceType3D> mtx_tex;
    #
    #  __global__ void copy_texture(float *dest, int rw)
    #  {
    #    int row   = blockIdx.x*blockDim.x + threadIdx.x;
    #    int col   = blockIdx.y*blockDim.y + threadIdx.y;
    #    int slice = blockIdx.z*blockDim.z + threadIdx.z;
    #    int tid = row + blockDim.x * gridDim.x * (col + slice * blockDim.y*gridDim.y);
    #    if (rw==0){
    #       float aux = dest[tid];
    #
    #       surf3Dwrite(aux, mtx_tex, slice * sizeof(float), col, row, cudaBoundaryModeTrap);
    #       float aux2 = dest[tid];
    #       surf3Dread(&aux2, mtx_tex, slice * sizeof(float), col, row, cudaBoundaryModeTrap);
    #       printf("(%d, %d, %d)[%d]:   %f, %f \\n", row, col, slice, tid, aux, aux2);
    #
    #    } else {
    #       float aux = 0;
    #       surf3Dread(&aux, mtx_tex, slice * sizeof(float), col, row, cudaBoundaryModeTrap);
    #       // printf("(%d, %d, %d)[%d] =  %f \\n", row, col, slice, tid, aux);
    #       dest[tid] = aux;
    #    }
    #  }
    #  """
    # # npoints = 32
    # A_cpu = np.zeros((shape[2], shape[1], shape[0]), order="C", dtype=np.float32)
    # A_cpu[:] = np.random.rand(shape[2], shape[1], shape[0])[:]
    # A_gpu = pycuda.gpuarray.to_gpu(A_cpu)
    # prec_str = dtype_to_ctype(np.float32)
    #
    # modW = SourceModule(myKernRW)
    # copy_texture = modW.get_function("copy_texture")
    # mtx_tex = modW.get_surfref("mtx_tex")
    # cuBlock = (8, 8, 8)
    # if cuBlock[0] > max(shape):
    #     cuBlock = (shape[2], shape[1], shape[0])
    # cuGrid = (
    #     shape[2] // cuBlock[0] + 1 * (shape[2] % cuBlock[0] != 0),
    #     shape[1] // cuBlock[1] + 1 * (shape[1] % cuBlock[1] != 0),
    #     shape[0] // cuBlock[2] + 1 * (shape[0] % cuBlock[2] != 0),
    # )
    # copy_texture.prepare("Pi")  # ,texrefs=[mtx_tex])
    # A_gpu2 = pycuda.gpuarray.zeros_like(A_gpu)  # To initialize surface with zeros
    # A_gpu3 = pycuda.gpuarray.zeros_like(A_gpu)  # To initialize surface with zeros
    # cudaArray = pycuda.driver.gpuarray_to_array(A_gpu2, "C", allowSurfaceBind=True)
    # A_cpu = A_gpu.get()  # To remember original array
    # mtx_tex.set_array(cudaArray)
    # copy_texture.prepared_call(
    #     cuGrid, cuBlock, A_gpu.gpudata, np.int32(0)
    # )  # Write random array
    # torch.cuda.synchronize()
    # print("\n\n")
    # copy_texture.prepared_call(
    #     cuGrid, cuBlock, A_gpu3.gpudata, np.int32(1)
    # )
    #
    # assert np.sum(np.abs(A_gpu3.get() - A_cpu)) == np.array(0, dtype=np.float32)
    #
    # mtx_srf = mod.get_surfref("mtx_srf")
    # # mtx_tex2 = mod.get_texref("mtx_tex2")
    # from pycuda.gpuarray import GPUArray
    # A_cpu = np.zeros((shape[2], shape[1], shape[0]), dtype=np.float32)
    # # A_cpu = np.zeros(self.conf.shape, dtype=np.float32)
    # # A_cpu[:] = np.random.rand(*self.conf.shape)
    # A_cpu[:] = np.random.rand(*(shape[2], shape[1], shape[0]))
    # A_gpu = pycuda.gpuarray.to_gpu(A_cpu)
    #
    # # A_gpu2 = pycuda.gpuarray.zeros_like(A_gpu)
    #
    # cudaArray = pycuda.driver.gpuarray_to_array(A_gpu, order="C", allowSurfaceBind=True)
    #
    # mtx_srf.set_array(cudaArray)
    # # mtx_tex2.set_array(cudaArray)
    # # mtx_srf.set_array(self.gpu_data)
    #
    # copy_texture = mod.get_function("copy_texture")
    # mtx_tex = mod.get_texref("mtx_tex")
    # mtx_tex.set_array(self.gpu_data)
    #
    # dest = np.zeros(shape, dtype=np.float32, order="C")
    # copy_texture(pycuda.driver.Out(dest), np.int64(self.gpu_data.handle), block=(shape[2], shape[1], shape[0]), texrefs=[mtx_tex])
    #
    # print()


class GPUArrayCollection:

    def __init__(self, device, bprint_allocated_memory=False):
        # torch.set_printoptions(precision=2)
        self.device = torch.device(device)
        self.last_allocated_memory = 0
        self.bprint_allocated_memory = bprint_allocated_memory
        self.registered_buffers = []

    def izeros(self, shape) -> torch.Tensor:
        return torch.zeros(shape, dtype=torch.int32, device=self.device)

    def fzeros(self, shape) -> torch.Tensor:
        return torch.zeros(shape, dtype=torch.float32, device=self.device)

    def frand(self, shape) -> torch.Tensor:
        return torch.rand(shape, dtype=torch.float32, device=self.device)

    @staticmethod
    def to_dataframe(tensor: torch.Tensor):
        return pd.DataFrame(tensor.cpu().numpy())

    def print_allocated_memory(self, naming='', f=10**9):
        if self.bprint_allocated_memory:
            last = self.last_allocated_memory
            self.last_allocated_memory = now = torch.cuda.memory_allocated(0) / f
            diff = np.round((self.last_allocated_memory - last), 3)
            unit = 'GB'
            unit2 = 'GB'
            if self.last_allocated_memory < 0.1:
                now = now * 10 ** 3
                unit = 'MB'
            if diff < 0.1:
                diff = np.round((self.last_allocated_memory - last) * 10 ** 3, 1)
                unit2 = 'MB'
            now = np.round(now, 1)
            print(f"memory_allocated({naming}) = {now}{unit} ({'+' if diff >= 0 else ''}{diff}{unit2})")

    def unregister_registered_buffers(self):
        for rb in self.registered_buffers:
            rb.unregister()
