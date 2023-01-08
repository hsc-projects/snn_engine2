import numpy as np

from vispy import io
# from vispy.gloo.texture import should_cast_to_f32
# from vispy.color import get_colormap
# from vispy.gloo import IndexBuffer, VertexBuffer
from vispy.scene.visuals import Volume
# from vispy.visuals import Visual
from vispy.visuals.transforms import STTransform

from rendering import RenderedCudaObjectNode, RegisteredTexture3D


# noinspection PyAbstractClass
class ChemicalConcentrationVolumeVisual(RenderedCudaObjectNode):

    def __init__(self, data, network_shape, name, scene, view, device):

        scene.set_current()

        if data is None:
            # height, depth, width
            # z, y, x
            data = self._test_volume()
        elif isinstance(data, tuple):
            assert len(data) == 3

        self._obj: Volume = Volume(data, texture_format='r32f')

        super().__init__(name=name, subvisuals=[self._obj])

        self.unfreeze()
        self.transform = STTransform()

        self.transform.scale = (network_shape[0]/(self._obj._vol_shape[2]),
                                network_shape[1]/(self._obj._vol_shape[1]),
                                network_shape[2]/(self._obj._vol_shape[0]))
        self.transform.move((.5 * self.transform.scale[0],
                             .5 * self.transform.scale[1],
                             .5 * self.transform.scale[2] + 1.1))
        self.additive = 100
        self.freeze()

        view.add(self)
        scene._draw_scene()

        self.init_cuda_attributes(device)

    def init_cuda_arrays(self):
        buffer = self.buffer_id(self._obj._texture.id)
        self._gpu_array = RegisteredTexture3D(
            buffer, self._obj._vol_shape, self._cuda_device, cpu_data=self._obj._last_data)

    def toggle_visible(self):
        self.visible = not self.visible

    @staticmethod
    def _test_volume():
        return np.array(np.load(io.load_data_file('volume/stent.npz'))['arr_0'], dtype=np.float32)

    def add(self):
        print(self._gpu_array.tensor[0, 0, 0])
        self._gpu_array.tensor += self.additive
        self._gpu_array.cpy_tnsr2tex()
        self._obj.update()
