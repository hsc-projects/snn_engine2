from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from rendering import RegisteredGPUArray, GPUArrayConfig, GPUArrayCollection, RegisteredIBO, RegisteredVBO
from network.network_grid import NetworkGrid
from network.network_config import NetworkConfig
from network.network_state.state_tensor import StateTensor, StateRow, Sliders
from network.network_state.neuron_flags import NeuronFlags
from network.network_structures import NeuronTypes


class LocationGroupFlags(StateTensor):

    @dataclass(frozen=True)
    class Rows:

        sensory_input_type: StateRow = StateRow(0, [-1, 1])
        b_thalamic_input: StateRow = StateRow(1, [0, 1])
        b_sensory_group: StateRow = StateRow(2, [0, 1])
        b_sensory_input: StateRow = StateRow(3, [0, 1])
        b_output_group: StateRow = StateRow(4, [0, 1])
        output_type: StateRow = StateRow(5, [-1, 1])
        b_monitor_group_firing_count: StateRow = StateRow(6, [0, 1])
        # inh_idx_start: StateRow = StateRow(7)

        def __len__(self):
            return 7

    def __init__(self, n_groups, device, select_ibo, grid: NetworkGrid, N_flags: NeuronFlags):
        self.group_ids = None
        self._rows = self.Rows()
        self._G = n_groups

        super().__init__(shape=(len(self._rows), n_groups), dtype=torch.int32, device=device)

        self.selected_array = RegisteredIBO(select_ibo, shape=(self._G+1, 1), device=device)

        # self.group_numbers_gpu: Optional[torch.Tensor] = None
        self.group_indices = None
        self.set_tensor(grid, N_flags)

    def set_tensor(self, grid: NetworkGrid, N_flags: NeuronFlags):
        thalamic_input_arr = torch.zeros(self._G)
        thalamic_input_arr[: int(self._G/2)] = 1
        # self.b_thalamic_input = thalamic_input_arr
        self.b_thalamic_input = 0
        # self.thalamic_input = 1

        self.b_sensory_group = torch.from_numpy(grid.sensory_group_mask).to(self._cuda_device)
        self.sensory_input_type = -1

        self.group_ids = (torch.arange(self._G).to(device=self._cuda_device)
                          .reshape((self._G, 1)))

        # self.b_monitor_group_firing_count[67] = 1
        # self.b_monitor_group_firing_count[68] = 1
        # self.b_monitor_group_firing_count[69] = 1
        # self.b_monitor_group_firing_count[123] = 1
        # self.b_monitor_group_firing_count[125] = 1
        self.b_monitor_group_firing_count = 1

        self.group_indices = self._set_group_indices(N_flags)

    @property
    def selected(self):
        # noinspection PyUnresolvedReferences
        return (self.selected_array.tensor != self._G).flatten()[: self._G]

    @selected.setter
    def selected(self, mask):
        self.selected_array.tensor[:self._G] = torch.where(mask.reshape((self._G, 1)), self.group_ids, self._G)

    @property
    def sensory_input_type(self):
        return self._tensor[self._rows.sensory_input_type.index, :]

    @sensory_input_type.setter
    def sensory_input_type(self, v):
        self._tensor[self._rows.sensory_input_type.index, :] = v

    @property
    def b_thalamic_input(self):
        return self._tensor[self._rows.b_thalamic_input.index, :]

    @b_thalamic_input.setter
    def b_thalamic_input(self, v):
        self._tensor[self._rows.b_thalamic_input.index, :] = v

    @property
    def b_sensory_group(self):
        return self._tensor[self._rows.b_sensory_group.index, :]

    @b_sensory_group.setter
    def b_sensory_group(self, v):
        self._tensor[self._rows.b_sensory_group.index, :] = v

    @property
    def b_output_group(self):
        return self._tensor[self._rows.b_output_group.index, :]

    @b_output_group.setter
    def b_output_group(self, v):
        self._tensor[self._rows.b_output_group.index, :] = v

    @property
    def output_type(self):
        return self._tensor[self._rows.output_type.index, :]

    @output_type.setter
    def output_type(self, v):
        self._tensor[self._rows.output_type.index, :] = v

    @property
    def b_sensory_input(self):
        return self._tensor[self._rows.b_sensory_input.index, :]

    @b_sensory_input.setter
    def b_sensory_input(self, v):
        self._tensor[self._rows.b_sensory_input.index, :] = v

    @property
    def b_monitor_group_firing_count(self):
        return self._tensor[self._rows.b_monitor_group_firing_count.index, :]

    @b_monitor_group_firing_count.setter
    def b_monitor_group_firing_count(self, v):
        self._tensor[self._rows.b_monitor_group_firing_count.index, :] = v

    def _set_group_indices(self, N_flags: NeuronFlags):
        indices = torch.zeros((self._G, 4), dtype=torch.int32).to(self._cuda_device) - 1
        for ntype in NeuronTypes:
            for g in range(self._G):
                ids = N_flags.id[
                    ((N_flags.type == ntype)
                     & (N_flags.group == g))]
                col = 2 * (ntype - 1)
                if len(ids) > 0:
                    indices[g, col] = ids[0]
                    indices[g, col + 1] = ids[-1]
        return indices

    def select(self, mask):
        self.selected = mask
        return self.group_ids[mask]


class LocationGroupProperties(StateTensor):

    @dataclass(frozen=True)
    class Rows:

        thalamic_inh_input_current: int = 0
        thalamic_exc_input_current: int = 1
        sensory_input_current0: int = 2
        sensory_input_current1: int = 3

        def __len__(self):
            return 4

    def __init__(self, n_groups, device, config, grid: NetworkGrid):
        self._rows = self.Rows()
        self._G = n_groups
        super().__init__(shape=(len(self._rows), n_groups), device=device, dtype=torch.float32)

        self.input_face_colors: Optional[torch.Tensor] = None
        self.output_face_colors: Optional[torch.Tensor] = None

        self.set_tensor(config)

        self.spin_box_sliders = Sliders(self.Rows())

    def set_tensor(self, config: NetworkConfig):

        self.thalamic_inh_input_current = config.InitValues.ThalamicInput.inh_current
        self.thalamic_exc_input_current = config.InitValues.ThalamicInput.exc_current
        self.sensory_input_current0 = config.InitValues.SensoryInput.input_current0
        self.sensory_input_current1 = config.InitValues.SensoryInput.input_current1

    @property
    def thalamic_inh_input_current(self):
        return self._tensor[self._rows.thalamic_inh_input_current, :]

    @thalamic_inh_input_current.setter
    def thalamic_inh_input_current(self, v):
        self._tensor[self._rows.thalamic_inh_input_current, :] = v

    @property
    def thalamic_exc_input_current(self):
        return self._tensor[self._rows.thalamic_exc_input_current, :]

    @thalamic_exc_input_current.setter
    def thalamic_exc_input_current(self, v):
        self._tensor[self._rows.thalamic_exc_input_current, :] = v

    @property
    def sensory_input_current0(self):
        return self._tensor[self._rows.sensory_input_current0, :]

    @sensory_input_current0.setter
    def sensory_input_current0(self, v):
        self._tensor[self._rows.sensory_input_current0, :] = v

    @property
    def sensory_input_current1(self):
        return self._tensor[self._rows.sensory_input_current1, :]

    @sensory_input_current1.setter
    def sensory_input_current1(self, v):
        self._tensor[self._rows.sensory_input_current1, :] = v


class G2GInfoArrays(GPUArrayCollection):

    float_arrays_list = [
        'G_distance',
        'G_avg_weight_inh',
        'G_avg_weight_exc',
    ]

    int_arrays_list = [
        'G_delay_distance',
        'G_stdp_config0',
        'G_stdp_config1',
        'G_syn_count_inh',
        'G_syn_count_exc',
        'G_rep'
    ]

    def __init__(self, network_config: NetworkConfig, group_ids, G_flags: LocationGroupFlags,
                 G_pos,
                 device, bprint_allocated_memory):
        super().__init__(device=device, bprint_allocated_memory=bprint_allocated_memory)

        self._config: NetworkConfig = network_config

        self.group_ids: torch.Tensor = group_ids
        self.G_flags: LocationGroupFlags = G_flags

        self.G_distance, self.G_delay_distance = self._G_delay_distance(network_config, G_pos)

        self.G_rep = torch.sort(self.G_delay_distance, dim=1, stable=True).indices.int()

        self.G_stdp_config0 = self.izeros(self.shape)
        self.G_stdp_config1 = self.izeros(self.shape)

        self.G_avg_weight_inh = self.fzeros(self.shape)
        self.G_avg_weight_exc = self.fzeros(self.shape)
        self.G_syn_count_inh = self.izeros(self.shape)
        self.G_syn_count_exc = self.izeros(self.shape)

    @property
    def shape(self):
        return self.G_distance.shape

    # noinspection PyPep8Naming
    @staticmethod
    def _G_delay_distance(network_config: NetworkConfig, G_pos: RegisteredVBO):
        G_pos_distance = torch.cdist(G_pos.tensor[: -1], G_pos.tensor[:-1])
        return G_pos_distance, ((network_config.D - 1) * G_pos_distance / G_pos_distance.max()).round().int()

    def _stdp_distance_based_config(self, target_group, anti_target_group, target_config: torch.Tensor):
        distance_to_target_group = self.G_distance[:, target_group]
        distance_to_anti_target_group = self.G_distance[:, anti_target_group]

        xx0 = distance_to_target_group.reshape(self._config.G, 1).repeat(1, self._config.G)
        xx1 = distance_to_anti_target_group.reshape(self._config.G, 1).repeat(1, self._config.G)

        mask0 = xx0 < distance_to_target_group
        mask1 = xx0 <= xx1

        mask = mask0 & mask1

        target_config[mask] = 1
        target_config[~mask] = -1
        target_config[(xx0 == distance_to_target_group) & mask1] = 0

    def set_active_output_groups(self, output_groups=None, ):
        if output_groups is None:
            output_groups = self.active_output_groups()
        assert len(output_groups) == 2
        output_group_types = self.G_flags.output_type[output_groups].type(torch.int64)

        group0 = output_groups[output_group_types == 0].item()
        group1 = output_groups[output_group_types == 1].item()

        self._stdp_distance_based_config(group0, anti_target_group=group1, target_config=self.G_stdp_config0)
        # noinspection PyUnusedLocal
        b = self.to_dataframe(self.G_stdp_config0)

        self._stdp_distance_based_config(group1, anti_target_group=group0, target_config=self.G_stdp_config1)

    def active_output_groups(self):
        return self.group_ids[self.G_flags.b_output_group.type(torch.bool)]
