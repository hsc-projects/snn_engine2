import numpy as np
from typing import Dict, Optional, Union

from network.network_config import (
    NetworkConfig,
    PlottingConfig
)
from network.gpu.simulation import NetworkSimulationGPU
from network.gpu.neurons import NeuronRepresentation
from network.gpu.synapses import SynapseRepresentation
# from network.gpu.chemicals import ChemicalRepresentation
from network.chemical_config import ChemicalConfigCollection, DefaultChemicals
from .network_structures import (
    NeuronTypes,
    NeuronTypeGroup,
    NeuronTypeGroupConnection
)
from network.gpu.visualized_elements.boxes import (
    InputGroups,
    OutputGroups,
    SelectorBox,
    OuterGrid
)
from network.gpu.visualized_elements import (
    GroupFiringCountsPlot
)


from signaling import SignalCollection

# from engine import App


# noinspection PyPep8Naming
# class NetworkCPUArrays:
#
#     def __init__(self, config: NetworkConfig, gpu: NetworkGPUArrays):
#
#         self._config = config
#         self.cpp_cuda_backend = gpu
#
#         self.N_rep: np.array = gpu.synapse_arrays.N_rep.cpu()
#         # self.N_G: np.array = gpu.N_G.cpu()
#
#         # self.group_indices: np.array = gpu.group_indices.cpu()
#
#         self.N_rep_groups: np.array = self.cpp_cuda_backend.synapse_arrays.N_rep_groups_cpu
#
#     @staticmethod
#     def to_dataframe(tensor: torch.Tensor):
#         return pd.DataFrame(tensor.numpy())


class SpikingNeuralNetwork:

    # noinspection PyPep8Naming
    def __init__(self, config, engine):

        self.T = config.T
        self.network_config: NetworkConfig = config.network
        self._plotting_config: PlottingConfig = config.plotting
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

        self.registered_buffers = []

        self.neurons = NeuronRepresentation(
            engine,
            config.network,
            type_group_dct=self.type_group_dct,
            type_group_conn_dct=self.type_group_conn_dict,
            device=config.device,
            T=self.T,
            plotting_config=self._plotting_config)

        self.registered_buffers += self.neurons.registered_buffers

        self.synapse_arrays = SynapseRepresentation(
            scene=engine.main_window.scene_3d,
            view=engine.main_window.scene_3d.network_view,
            neurons=self.neurons,
            device=config.device, shapes=self.neurons.data_shapes)

        self.chemical_concentrations: Optional[Union[ChemicalConfigCollection,
                                                     DefaultChemicals]] = (
            config.network.chemical_configs if config.network.chemical_configs is not None
            else ChemicalConfigCollection())

        self.chemical_concentrations.super_init(
            network_shape=self.network_config.N_pos_shape,
            scene=engine.main_window.scene_3d,
            view=engine.main_window.scene_3d.network_view,
            device=config.device)

        self.simulation_gpu: Optional[NetworkSimulationGPU] = None
        # self.CPU: Optional[NetworkCPUArrays] = None

        self.outer_grid: Optional[OuterGrid] = None

        self.group_firing_counts_multiplot: Optional[GroupFiringCountsPlot] = None
        self.group_firing_counts_plot_single0: Optional[GroupFiringCountsPlot] = None
        self.group_firing_counts_plot_single1: Optional[GroupFiringCountsPlot] = None

        self.input_groups: Optional[InputGroups] = None
        self.output_groups: Optional[OutputGroups] = None

        self.validate()

        self.initialize_GPU_arrays(config.device, engine)

        if self.simulation_gpu.synapse_arrays.pre_synaptic_rep_initialized is False:
            raise AssertionError

    @property
    def plotting_config(self):
        return self._plotting_config

    def validate(self):
        NeuronTypeGroup.validate(self.type_group_dct, N=self.network_config.N)
        NeuronTypeGroupConnection.validate(self.type_group_conn_dict, S=self.network_config.S)

    def add_input_groups(self, scene, view):
        if self.input_groups is not None:
            raise AttributeError
        self.input_groups = InputGroups(
            scene=scene, view=view,
            data=np.array([0, 1, 0], dtype=np.int32),
            pos=np.array([[int(self.network_config.N_pos_shape[0]/2 + 1) * self.network_config.grid.unit_shape[1],
                           0.,
                           self.network_config.N_pos_shape[2] - self.network_config.grid.unit_shape[2]]]),
            network=self,
            state_colors_attr='input_face_colors',
            compatible_groups=self.network_config.sensory_groups,
        )

        self.input_groups.init_cuda_attributes(
            self.simulation_gpu.device, self.neurons.G_flags, self.neurons.G_props)
        self.input_groups.src_weight = self.network_config.InitValues.Weights.SensorySource

        self.registered_buffers += self.input_groups.registered_buffers

    def add_output_groups(self, scene, view):
        if self.output_groups is not None:
            raise AttributeError
        shape = self.neurons._neuron_visual._shape
        self.output_groups = OutputGroups(
            scene=scene, view=view,
            data=np.array([0, -1, 1], dtype=np.int32),
            pos=np.array([[int(shape[0]/2 + 1) * self.network_config.grid.unit_shape[1],
                           shape[1] - self.network_config.grid.unit_shape[1],
                           shape[2] - self.network_config.grid.unit_shape[2]]]),
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

        self.output_groups.init_cuda_attributes(self.simulation_gpu.device,
                                                self.neurons.G_flags,
                                                self.neurons.G_props)
        self.registered_buffers += self.output_groups.registered_buffers

    def add_selector_box(self, scene, view):
        selector_box = SelectorBox(scene, view, self.network_config, self.network_config.grid,
                                   self.simulation_gpu.device,
                                   self.neurons.G_flags, self.neurons.G_props)
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
        self.simulation_gpu.update()

    # noinspection PyPep8Naming
    def initialize_GPU_arrays(self, device, engine, init_default_sim=False,
                              init_default_sim_with_syn_post_init=False):

        if init_default_sim is False:
            assert init_default_sim_with_syn_post_init is False

        self.outer_grid: OuterGrid = OuterGrid(
            view=engine.main_window.scene_3d.network_view,
            shape=self.network_config.N_pos_shape,
            segments=self.network_config.grid_segmentation)

        engine.set_main_context_as_current()

        if init_default_sim is True:
            self.simulation_gpu = NetworkSimulationGPU.from_snn(self, engine=engine, device=device)
            self.registered_buffers += self.simulation_gpu.registered_buffers
            if init_default_sim_with_syn_post_init is True:
                self.simulation_gpu._post_synapse_mod_init()

    def interface_single_neurons(self, engine):
        pass

    def unregister_registered_buffers(self):
        for rb in self.registered_buffers:
            rb.unregister()
        self.synapse_arrays.unregister_registered_buffers()

