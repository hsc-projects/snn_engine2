import numpy as np
import torch

# from network.gpu.plotting import PlottingGPUArrays
# from network.gpu.synapses import SynapseRepresentation
from network.network_array_shapes import NetworkArrayShapes
from network.network_config import NetworkConfig, PlottingConfig
from network.network_state import (
    LocationGroupFlags,
    G2GInfoArrays,
    LocationGroupProperties, MultiModelNeuronStateTensor,
    NeuronFlags
)
from network.network_structures import NeuronTypeGroup, NeuronTypes, NeuronTypeGroupConnection
from network.network_grid import NetworkGrid
from network.gpu.visualized_elements import NeuronVisual
from network.gpu.visualized_elements.boxes import SelectedGroups

# noinspection PyUnresolvedReferences
from network.gpu.cpp_cuda_backend import (
    snn_construction_gpu,
    snn_simulation_gpu,
    GPUArrayConfig,
    RegisteredVBO,
    GPUArrayCollection
)


# noinspection PyPep8Naming
class NeuronRepresentation(GPUArrayCollection):

    def __init__(self,
                 engine,
                 config: NetworkConfig,
                 type_group_dct: dict,
                 type_group_conn_dct: dict,
                 device: int,
                 T,
                 plotting_config: PlottingConfig,
                 ):

        super().__init__(device=device, bprint_allocated_memory=config.N > 1000)

        self._config: NetworkConfig = config
        self._plotting_config: PlottingConfig = plotting_config
        self._type_group_dct = type_group_dct
        self._type_group_conn_dct = type_group_conn_dct

        self._neuron_visual = NeuronVisual(
            engine.main_window.scene_3d, engine.main_window.scene_3d.network_view,
            device,
            self._config, self._config.grid.segmentation, self.type_groups)

        self.registered_buffers += self._neuron_visual.registered_buffers

        self.data_shapes = NetworkArrayShapes(config=self._config, T=T,
                                              plotting_config=self._plotting_config,
                                              n_neuron_types=len(NeuronTypes))

        self.curand_states = self._curand_states()

        self.N_flags: NeuronFlags = NeuronFlags(n_neurons=self._config.N, device=self.device)

        (self.G_neuron_counts,
         self.G_neuron_typed_ccount) = self._N_G_and_G_neuron_counts_1of2(self.data_shapes, self._config.grid)

        self.group_indices = None

        self.selected_group_boxes = SelectedGroups(
            scene=engine.main_window.scene_3d,
            view=engine.main_window.scene_3d.network_view,
            network_config=self._config, grid=self._config.grid,
            connect=np.zeros((self._config.G + 1, 2)) + self._config.G,
            device=device,
        )

        self.G_pos: RegisteredVBO = RegisteredVBO(self.selected_group_boxes.vbo, self.data_shapes.G_pos, self.device)
        self.registered_buffers.append(self.G_pos)
        self.G_flags = LocationGroupFlags(self._config.G, device=self.device, grid=self._config.grid,
                                          select_ibo=self.selected_group_boxes.ibo, N_flags=self.N_flags)
        self.registered_buffers.append(self.G_flags.selected_array)

        self.g2g_info_arrays = G2GInfoArrays(self._config, self.G_flags.group_ids,
                                             self.G_flags, self.G_pos,
                                             device=device, bprint_allocated_memory=self.bprint_allocated_memory)

        self._G_neuron_counts_2of2(self.g2g_info_arrays.G_delay_distance, self.G_neuron_counts)
        self.G_group_delay_counts = self._G_group_delay_counts(self.data_shapes.G_delay_counts,
                                                               self.g2g_info_arrays.G_delay_distance)

        self.G_props = LocationGroupProperties(self._config.G, device=self.device,
                                               config=self._config, grid=self._config.grid)

        self.selected_group_boxes.init_cuda_attributes(G_flags=self.G_flags,
                                                       G_props=self.G_props)
        self.registered_buffers += self.selected_group_boxes.registered_buffers

        self.N_states = MultiModelNeuronStateTensor(
            self._config.N, device=self.device, flag_tensor=self.N_flags)

    @property
    def type_groups(self) -> list[NeuronTypeGroup]:
        return list(self._type_group_dct.values())

    def _curand_states(self):
        cu = snn_construction_gpu.CuRandStates(self._config.N).ptr()
        self.print_allocated_memory('curand_states')
        return cu

    def _G_group_delay_counts(self, shape, G_delay_distance):
        G_group_delay_counts = self.izeros(shape)
        for d in range(self._config.D):
            G_group_delay_counts[:, d + 1] = (G_group_delay_counts[:, d] + G_delay_distance.eq(d).sum(dim=1))
        return G_group_delay_counts

    def _G_neuron_counts_2of2(self, G_delay_distance, G_neuron_counts):
        snn_construction_gpu.fill_G_neuron_count_per_delay(
            S=self._config.S, D=self._config.D, G=self._config.G,
            G_delay_distance=G_delay_distance.data_ptr(),
            G_neuron_counts=G_neuron_counts.data_ptr())
        self.validate_G_neuron_counts()

    def _N_G_and_G_neuron_counts_1of2(self, shapes: NetworkArrayShapes, grid: NetworkGrid):
        # N_G = self.izeros(shapes.N_G)
        # t_neurons_ids = torch.arange(self.N_G.shape[0], device='cuda')  # Neuron Id
        for g in self.type_groups:
            self.N_flags.type[g.start_idx:g.end_idx + 1] = g.ntype.value  # Set Neuron Type

        # rows[0, 1]: inhibitory count, excitatory count,
        # rows[2 * D]: number of neurons per delay (post_synaptic type: inhibitory, excitatory)
        G_neuron_counts = self.izeros(shapes.G_neuron_counts)
        snn_construction_gpu.fill_N_flags_group_id_and_G_neuron_count_per_type(
            N=self._config.N, G=self._config.G,
            N_pos=self._neuron_visual.gpu_array.data_ptr(),
            N_pos_shape=self._config.N_pos_shape,
            N_flags=self.N_flags.data_ptr(),
            G_shape=grid.segmentation,
            G_neuron_counts=G_neuron_counts.data_ptr(),
            N_flags_row_type=self.N_flags.rows.type.index,
            N_flags_row_group=self.N_flags.rows.group.index
        )

        G_neuron_typed_ccount = self.izeros((2 * self._config.G + 1))
        G_neuron_typed_ccount[1:] = G_neuron_counts[: 2, :].ravel().cumsum(dim=0)
        self.N_flags.validate(self._neuron_visual, self._neuron_visual.gpu_array)
        return G_neuron_counts, G_neuron_typed_ccount

    def validate_G_neuron_counts(self):
        D, G = self._config.D, self._config.G
        max_ntype = 0
        for ntype_group in self.type_groups:
            if self.G_neuron_counts[ntype_group.ntype - 1, :].sum() != len(ntype_group):
                raise AssertionError
            max_ntype = max(max_ntype, ntype_group.ntype)

        for ntype_group in self.type_groups:
            min_row = max_ntype + D * (ntype_group.ntype - 1)
            max_row = min_row + D
            expected_result = (self.izeros(G) + 1) * len(ntype_group)
            if ((self.G_neuron_counts[min_row: max_row, :].sum(dim=0)
                 - expected_result).sum() != 0):
                print(self.G_neuron_counts)
                raise AssertionError

    @property
    def type_group_conns(self) -> list[NeuronTypeGroupConnection]:
        return list(self._type_group_conn_dct.values())

    def select_groups(self, mask):
        return self.G_flags.group_ids[mask]

    @property
    def active_sensory_groups(self):
        return self.select_groups(self.G_flags.b_sensory_input.type(torch.bool))

    @property
    def active_output_groups(self):
        return self.select_groups(self.G_flags.b_output_group.type(torch.bool))
