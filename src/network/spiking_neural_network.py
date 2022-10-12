import numpy as np
import pandas as pd
import torch
from typing import Dict, Optional

from .network_config import (
    NetworkConfig,
    PlottingConfig
)
from .network_array_shapes import NetworkArrayShapes
from network.gpu_arrays.network_gpu_arrays import NetworkGPUArrays
from .network_structures import (
    NeuronTypes,
    NeuronTypeGroup,
    NeuronTypeGroupConnection
)
from network.visualized_elements.boxes import (
    SelectedGroups,
    InputGroups,
    OutputGroups,
    SelectorBox,
    GroupInfo,
    OuterGrid
)
from network.visualized_elements import (
    Neurons,
    VoltageMultiPlot,
    FiringScatterPlot,
    # GroupFiringCountsPlot0,
    GroupFiringCountsPlot
)
# from .network_state.izhikevich_model import IzhikevichModel
from .network_grid import NetworkGrid


from signaling import SignalCollection

# from engine import App


# noinspection PyPep8Naming
# class NetworkCPUArrays:
#
#     def __init__(self, config: NetworkConfig, gpu_arrays: NetworkGPUArrays):
#
#         self._config = config
#         self.gpu = gpu_arrays
#
#         self.N_rep: np.array = gpu_arrays.synapse_arrays.N_rep.cpu()
#         # self.N_G: np.array = gpu_arrays.N_G.cpu()
#
#         # self.group_indices: np.array = gpu_arrays.group_indices.cpu()
#
#         self.N_rep_groups: np.array = self.gpu.synapse_arrays.N_rep_groups_cpu
#
#     @staticmethod
#     def to_dataframe(tensor: torch.Tensor):
#         return pd.DataFrame(tensor.numpy())


class SpikingNeuralNetwork:

    # noinspection PyPep8Naming
    def __init__(self, config, app):

        # RenderedObjectNode._grid_unit_shape = network_config.grid_unit_shape

        self.T = config.T
        self.network_config: NetworkConfig = config.network
        self._plotting_config: PlottingConfig = config.plotting
        self.grid: NetworkGrid = config.network.grid
        self.signal_collection = SignalCollection()

        print('\n', self.network_config, '\n')
        self.max_batch_size_mb = config.max_batch_size_mb

        self.type_group_dct: Dict[int, NeuronTypeGroup] = {}
        self.type_group_conn_dict: Dict[tuple[int, int], NeuronTypeGroupConnection] = {}
        self.next_group_id = 0

        g_inh = self.add_type_group(count=int(.2 * self.network_config.N), neuron_type=NeuronTypes.INHIBITORY)
        g_exc = self.add_type_group(count=self.network_config.N - len(g_inh), neuron_type=NeuronTypes.EXCITATORY)

        self.add_type_group_conn(
            g_inh, g_exc,
            w0=self.network_config.InitValues.Weights.Inh2Exc,
            exp_syn_counts=self.network_config.S)
        c_exc_inh = self.add_type_group_conn(
            g_exc, g_inh,
            w0=self.network_config.InitValues.Weights.Exc2Inh,
            exp_syn_counts=max(int((len(g_inh) / self.network_config.N) * self.network_config.S), 1))
        self.add_type_group_conn(
            g_exc, g_exc,
            w0=self.network_config.InitValues.Weights.Exc2Exc,
            exp_syn_counts=self.network_config.S - len(c_exc_inh))

        self._neurons = Neurons(self.network_config, self.grid.segmentation, self.type_groups)

        self.GPU: Optional[NetworkGPUArrays] = None
        # self.CPU: Optional[NetworkCPUArrays] = None

        self.outer_grid: Optional[OuterGrid] = None
        self.voltage_plot: Optional[VoltageMultiPlot] = None
        self.firing_scatter_plot: Optional[FiringScatterPlot] = None

        self.group_firing_counts_multiplot: Optional[GroupFiringCountsPlot] = None
        self.group_firing_counts_plot_single0: Optional[GroupFiringCountsPlot] = None
        self.group_firing_counts_plot_single1: Optional[GroupFiringCountsPlot] = None

        self.input_groups: Optional[InputGroups] = None
        self.output_groups: Optional[OutputGroups] = None
        self.selected_group_boxes: Optional[SelectedGroups] = None

        self.group_info_mesh: Optional[GroupInfo] = None

        self._all_rendered_objects_initialized = False

        self.data_shapes = NetworkArrayShapes(config=self.network_config, T=self.T,
                                              # n_N_states=model.__len__(),
                                              plotting_config=self.plotting_config,
                                              n_neuron_types=len(NeuronTypes))
        self.registered_buffers = []
        self.validate()

        self.initialize_GPU_arrays(config.device, app)

        if self.GPU.synapse_arrays.pre_synaptic_rep_initialized is False:
            raise AssertionError

    @property
    def plotting_config(self):
        return self._plotting_config

    def validate(self):
        NeuronTypeGroup.validate(self.type_group_dct, N=self.network_config.N)
        NeuronTypeGroupConnection.validate(self.type_group_conn_dict, S=self.network_config.S)

    @property
    def type_groups(self):
        return self.type_group_dct.values()

    def add_input_groups(self, scene, view):
        if self.input_groups is not None:
            raise AttributeError
        scene.set_current()
        self.input_groups = InputGroups(
            data=np.array([0, 1, 0], dtype=np.int32),
            pos=np.array([[int(self.network_config.N_pos_shape[0]/2 + 1) * self.grid.unit_shape[1],
                           0.,
                           self.network_config.N_pos_shape[2] - self.grid.unit_shape[2]]]),
            network=self,
            state_colors_attr='input_face_colors',
            compatible_groups=self.network_config.sensory_groups,
        )
        view.add(self.input_groups)
        scene._draw_scene()
        self.input_groups.init_cuda_attributes(self.GPU.device, self.GPU.G_flags, self.GPU.G_props)
        self.input_groups.src_weight = self.network_config.InitValues.Weights.SensorySource

        self.registered_buffers += self.input_groups.registered_buffers

    def add_output_groups(self, scene, view):
        if self.output_groups is not None:
            raise AttributeError
        scene.set_current()

        self.output_groups = OutputGroups(
            data=np.array([0, -1, 1], dtype=np.int32),
            pos=np.array([[int(self._neurons._shape[0]/2 + 1) * self.grid.unit_shape[1],
                           self._neurons._shape[1] - self.grid.unit_shape[1],
                           self._neurons._shape[2] - self.grid.unit_shape[2]]]),
            state_colors_attr='output_face_colors',
            network=self,
            data_color_coding=np.array([
                [1., 0., 0., .6],
                [0., 1., 0., .6],
                # [0., 0., 0., .0],
            ]),
            compatible_groups=self.network_config.output_groups,
            face_dir='+z',
        )
        view.add(self.output_groups)
        scene._draw_scene()

        self.output_groups.init_cuda_attributes(self.GPU.device, self.GPU.G_flags, self.GPU.G_props)
        self.registered_buffers += self.output_groups.registered_buffers

    def add_selector_box(self, scene, view):
        scene.set_current()
        selector_box = SelectorBox(self.network_config, self.grid, scene)
        view.add(selector_box)
        scene._draw_scene()
        selector_box.init_cuda_attributes(self.GPU.device, self.GPU.G_flags, self.GPU.G_props)
        self.registered_buffers += selector_box.registered_buffers
        return selector_box

    # noinspection PyPep8Naming
    def add_type_group(self, count, neuron_type):
        g = NeuronTypeGroup.from_count(self.next_group_id, count, self.network_config.S,
                                       neuron_type, self.type_group_dct)
        self.next_group_id += 1
        return g

    def add_type_group_conn(self, src, snk, w0, exp_syn_counts):
        c = NeuronTypeGroupConnection(src, snk, w0=w0, S=self.network_config.S,
                                      exp_syn_counts=exp_syn_counts,
                                      max_batch_size_mb=self.max_batch_size_mb,
                                      conn_dict=self.type_group_conn_dict)
        return c

    def update(self):
        self.GPU.update()

    # noinspection PyPep8Naming
    def initialize_GPU_arrays(self, device, engine):

        engine.set_main_context_as_current()

        engine.main_window.scene_3d.network_view.add(self._neurons)

        self.outer_grid: OuterGrid = OuterGrid(
            view=engine.main_window.scene_3d.network_view,
            shape=self.network_config.N_pos_shape,
            segments=self.network_config.grid_segmentation)

    def interface_single_neurons(self, engine):
        pass

    def unregister_registered_buffers(self):
        for rb in self.registered_buffers:
            rb.unregister()

