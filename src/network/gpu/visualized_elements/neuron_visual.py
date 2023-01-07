import numpy as np
import torch

from vispy.scene import visuals

from rendering import RenderedCudaObjectNode, RegisteredVBO
from geometry import grid_coordinates, validate_pos
from network.network_config import NetworkConfig
from network.network_grid import NetworkGrid
from network.network_structures import NeuronTypes


# noinspection PyAbstractClass
class NeuronVisual(RenderedCudaObjectNode):

    def __init__(self,
                 scene, view, device,
                 config: NetworkConfig, grid_segmentation: NetworkGrid, type_groups):

        scene.set_current()

        self._config = config
        self._shape = config.N_pos_shape
        if config.pos is None:
            self._config.pos = (np.random.rand(config.N, 3).astype(np.float32) * np.array(self._shape, dtype=np.float32))
            self._config.pos[self._config.pos == max(self._shape)] = config.pos[config.pos == max(self._shape)] * 0.999999
            validate_pos(self._config.pos, self._shape)

        self.grid_coord = grid_coordinates(self._config.pos, self._shape, grid_segmentation)

        self._type_groups = type_groups
        self.sort_pos()

        assert self._config.pos.shape[1] == 3
        assert len(self._config.pos.shape) == 2
        assert np.max(self._config.pos[:, 2]) < config.max_z

        self._obj: visuals.visuals.MarkersVisual = visuals.Markers()
        self._obj.set_data(self._config.pos,
                           face_color=(1, 1, 1, .3),
                           edge_color=(0, 0.02, 0.01, .5),
                           size=7, edge_width=1)

        super().__init__(name='Neurons', subvisuals=[self._obj])

        # noinspection PyTypeChecker
        self.set_gl_state('translucent', blend=True, depth_test=True)

        view.add(self)
        scene._draw_scene()

        self.init_cuda_attributes(device)

    def init_cuda_arrays(self):

        self._gpu_array = RegisteredVBO(self.vbo,
                                        (len(self._config.pos), self._config.vispy_scatter_plot_stride),
                                        self._cuda_device)
        for g in self._type_groups:
            if g.ntype == NeuronTypes.INHIBITORY:
                orange = torch.Tensor([1, .5, .2])
                self._gpu_array.tensor[g.start_idx:g.end_idx + 1, 7:10] = orange  # Inhibitory Neurons -> Orange
        self.registered_buffers.append(self._gpu_array)

    def sort_pos(self):
        """
        Sort neuron positions w.r.t. location-based groups and neuron types.
        """

        for g in self._type_groups:

            grid_pos = self.grid_coord[g.start_idx: g.end_idx + 1]

            p0 = grid_pos[:, 0].argsort(kind='stable')
            p1 = grid_pos[p0][:, 1].argsort(kind='stable')
            p2 = grid_pos[p0][p1][:, 2].argsort(kind='stable')

            self.grid_coord[g.start_idx: g.end_idx + 1] = grid_pos[p0][p1][p2]
            self._config.pos[g.start_idx: g.end_idx + 1] = \
                self._config.pos[g.start_idx: g.end_idx + 1][p0][p1][p2]
            if len(self._config.pos) <= 100:
                print('\n', self._config.pos[g.start_idx:g.end_idx+1])
        if len(self._config.pos) <= 100:
            print()

    @property
    def vbo_glir_id(self):
        # noinspection PyProtectedMember
        return self._obj._vbo.id

    @property
    def edge_color(self):
        return self._gpu_array.tensor[:, 3:7]

    @property
    def face_color(self):
        return self._gpu_array.tensor[:, 7:11]

    @property
    def pos(self):
        return self._gpu_array.tensor[:, 0:3]
