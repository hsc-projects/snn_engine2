import numpy as np
import torch

from network.gpu_arrays.plotting import PlottingGPUArrays
from network.gpu_arrays.synapses import SynapseRepresentation
from network.network_array_shapes import NetworkArrayShapes
from network.network_config import BufferCollection, NetworkConfig, PlottingConfig
from network.network_state import (
    LocationGroupFlags,
    G2GInfoArrays,
    LocationGroupProperties, MultiModelNeuronStateTensor,
    NeuronFlags
)
from network.network_structures import NeuronTypeGroup, NeuronTypeGroupConnection, NeuronTypes
from network.network_grid import NetworkGrid
from network.visualized_elements.neurons import Neurons

# noinspection PyUnresolvedReferences
from gpu import (
    snn_construction_gpu,
    snn_simulation_gpu,
    GPUArrayConfig,
    RegisteredVBO,
    GPUArrayCollection
)


# noinspection PyPep8Naming
class NetworkGPUArrays(GPUArrayCollection):

    def __init__(self,
                 config: NetworkConfig,
                 grid: NetworkGrid,
                 neurons: Neurons,
                 type_group_dct: dict,
                 type_group_conn_dct: dict,
                 device: int,
                 T: int,
                 shapes: NetworkArrayShapes,
                 plotting_config: PlottingConfig,
                 buffers: BufferCollection,
                 app,
                 ):

        super().__init__(device=device, bprint_allocated_memory=config.N > 1000)

        self._config: NetworkConfig = config
        self._plotting_config: PlottingConfig = plotting_config
        self._type_group_dct = type_group_dct
        self._type_group_conn_dct = type_group_conn_dct

        self.registered_buffers = []

        self.plotting_arrays = PlottingGPUArrays(plotting_config,
                                                 device=device, shapes=shapes, buffers=buffers,
                                                 bprint_allocated_memory=self.bprint_allocated_memory,
                                                 app=app)

        self.registered_buffers += self.plotting_arrays.registered_buffers
        self.curand_states = self._curand_states()
        self.N_pos: RegisteredVBO = self._N_pos(shape=shapes.N_pos, vbo=buffers.N_pos)

        self.N_flags: NeuronFlags = NeuronFlags(n_neurons=self._config.N, device=self.device)

        (self.G_neuron_counts,
         self.G_neuron_typed_ccount) = self._N_G_and_G_neuron_counts_1of2(shapes, grid, neurons)

        # self.N_flags.type = self.N_G[:, self._config.N_G_neuron_type_col]
        # self.N_flags.group = self.N_G[:, self._config.N_G_group_id_col]

        # self.neuron_ids = torch.arange(config.N).to(device=self.device)
        # self.group_ids = torch.arange(config.G).to(device=self.device)

        self.group_indices = None

        self.G_pos: RegisteredVBO = RegisteredVBO(buffers.selected_group_boxes_vbo, shapes.G_pos, self.device)
        self.registered_buffers.append(self.G_pos)

        self.G_flags = LocationGroupFlags(self._config.G, device=self.device, grid=grid,
                                          select_ibo=buffers.selected_group_boxes_ibo, N_flags=self.N_flags)
        self.registered_buffers.append(self.G_flags.selected_array)

        self.g2g_info_arrays = G2GInfoArrays(self._config, self.G_flags.group_ids,
                                             self.G_flags, self.G_pos,
                                             device=device, bprint_allocated_memory=self.bprint_allocated_memory)

        self._G_neuron_counts_2of2(self.g2g_info_arrays.G_delay_distance, self.G_neuron_counts)
        self.G_group_delay_counts = self._G_group_delay_counts(shapes.G_delay_counts,
                                                               self.g2g_info_arrays.G_delay_distance)

        self.G_props = LocationGroupProperties(self._config.G, device=self.device, config=self._config, grid=grid)

        self.synapse_arrays = SynapseRepresentation(
            config=config,
            type_groups=self.type_groups,
            type_group_conns=self.type_group_conns,
            CuRandStates_ptr=self.curand_states,
            G_neuron_counts=self.G_neuron_counts,
            G_group_delay_counts=self.G_group_delay_counts,
            G_neuron_typed_ccount=self.G_neuron_typed_ccount,
            g2g_info_arrays=self.g2g_info_arrays,
            N_pos=self.N_pos,
            N_flags=self.N_flags,
            G_flags=self.G_flags,
            G_props=self.G_props,
            device=device, shapes=shapes)

        self.N_states = MultiModelNeuronStateTensor(
            self._config.N, device=self.device, flag_tensor=self.N_flags)

        self.Fired = self.fzeros(self._config.N)
        self.last_Fired = self.izeros(self._config.N) - self._config.D
        self.Firing_times = self.fzeros(shapes.Firing_times)
        self.Firing_idcs = self.izeros(shapes.Firing_idcs)
        self.Firing_counts = self.izeros(shapes.Firing_counts)

        self.G_firing_count_hist = self.izeros((self._plotting_config.scatter_plot_length, self._config.G))

        self.G_stdp_config0 = self.izeros((self._config.G, self._config.G))
        self.G_stdp_config1 = self.izeros((self._config.G, self._config.G))

        self.Simulation = self._init_sim(T, plotting_config)

        if self._config.N >= 8000:
            self.synapse_arrays.swap_group_synapses(
                groups=torch.from_numpy(grid.forward_groups).to(device=self.device).type(torch.int64))

        self.synapse_arrays.actualize_N_rep_pre_synaptic_idx(shapes=shapes)

        # self.N_states.use_preset('RS', self.N_flags.select_by_groups(self._config.sensory_groups))
        self.N_states.izhikevich_neurons.use_preset('RS', self.N_flags.select_by_groups(self._config.sensory_groups))

        self.Simulation.set_stdp_config(0)

        self.synapse_arrays.N_rep_buffer = self.synapse_arrays.N_rep_buffer.reshape(shapes.N_rep)
        # self.N_rep_buffer[:] = self.N_rep_groups_cpu.to(self.device)

        self.Simulation.calculate_avg_group_weight()

        self.output_tensor = self.fzeros(6)

        n0 = self.G_neuron_typed_ccount[67]

        self.N_flags.model[n0 + 2] = 1
        self.N_flags.model[n0 + 3] = 1
        self.N_flags.model[n0 + 4] = 1
        self.N_flags.model[n0 + 5] = 1

        self.synapse_arrays.N_rep[0, n0] = n0 + 2
        self.synapse_arrays.N_rep[0, n0 + 1] = n0 + 3
        self.synapse_arrays.N_rep[1, n0] = n0 + 4
        self.synapse_arrays.N_rep[1, n0 + 1] = n0 + 5

        # self.N_weights[0, n0] = 1.5
        # self.N_weights[0, n0 + 1] = 1.5
        # self.N_weights[1, n0] = 1.5
        # self.N_weights[1, n0 + 1] = 1.5

        # self.N_flags.model[self.G_neuron_typed_ccount[67] + 2] = 1
        # self.N_flags.model[self.G_neuron_typed_ccount[67] + 3] = 1

        self.synapse_arrays.make_sensory_group(
            G_neuron_counts=self.G_neuron_counts, N_pos=self.N_pos, G_pos=self.G_pos,
            groups=self._config.sensory_groups, grid=grid)

    def _init_sim(self, T, plotting_config):

        sim = snn_simulation_gpu.SnnSimulation(
            N=self._config.N,
            G=self._config.G,
            S=self._config.S,
            D=self._config.D,
            T=T,
            n_voltage_plots=plotting_config.n_voltage_plots,
            voltage_plot_length=plotting_config.voltage_plot_length,
            voltage_plot_data=self.plotting_arrays.voltage.data_ptr(),
            voltage_plot_map=self.plotting_arrays.voltage_map.data_ptr(),
            n_scatter_plots=plotting_config.n_scatter_plots,
            scatter_plot_length=plotting_config.scatter_plot_length,
            scatter_plot_data=self.plotting_arrays.firings.data_ptr(),
            scatter_plot_map=self.plotting_arrays.firings_map.data_ptr(),
            curand_states_p=self.curand_states,
            N_pos=self.N_pos.data_ptr(),
            # N_G=self.N_G.data_ptr(),
            G_group_delay_counts=self.G_group_delay_counts.data_ptr(),
            G_flags=self.G_flags.data_ptr(),
            G_props=self.G_props.data_ptr(),
            N_rep=self.synapse_arrays.N_rep.data_ptr(),
            N_rep_buffer=self.synapse_arrays.N_rep_buffer.data_ptr(),
            N_rep_pre_synaptic=self.synapse_arrays.N_rep_pre_synaptic.data_ptr(),
            N_rep_pre_synaptic_idcs=self.synapse_arrays.N_rep_pre_synaptic_idcs.data_ptr(),
            N_rep_pre_synaptic_counts=self.synapse_arrays.N_rep_pre_synaptic_counts.data_ptr(),
            N_delays=self.synapse_arrays.N_delays.data_ptr(),
            N_flags=self.N_flags.data_ptr(),
            N_states=self.N_states.data_ptr(),
            N_weights=self.synapse_arrays.N_weights.data_ptr(),
            fired=self.Fired.data_ptr(),
            last_fired=self.last_Fired.data_ptr(),
            firing_times=self.Firing_times.data_ptr(),
            firing_idcs=self.Firing_idcs.data_ptr(),
            firing_counts=self.Firing_counts.data_ptr(),
            G_firing_count_hist=self.G_firing_count_hist.data_ptr(),
            G_stdp_config0=self.g2g_info_arrays.G_stdp_config0.data_ptr(),
            G_stdp_config1=self.g2g_info_arrays.G_stdp_config1.data_ptr(),
            G_avg_weight_inh=self.g2g_info_arrays.G_avg_weight_inh.data_ptr(),
            G_avg_weight_exc=self.g2g_info_arrays.G_avg_weight_exc.data_ptr(),
            G_syn_count_inh=self.g2g_info_arrays.G_syn_count_inh.data_ptr(),
            G_syn_count_exc=self.g2g_info_arrays.G_syn_count_exc.data_ptr(),
            L_winner_take_all_map=self.synapse_arrays.L_winner_take_all_map.data_ptr(),
            max_n_winner_take_all_layers=self._config.max_n_winner_take_all_layers,
            max_winner_take_all_layer_size=self._config.max_winner_take_all_layer_size
        )

        return sim

    # noinspection PyUnusedLocal
    def look_up(self, tuples, input_tensor, output_tensor=None, precision=6):
        if output_tensor is None:
            if len(self.output_tensor) != len(tuples):
                self.output_tensor = self.fzeros(len(tuples))
            output_tensor = self.output_tensor
        output_tensor[:] = torch.nan
        for i, e in enumerate(tuples):
            output_tensor[i] = input_tensor[e]
        print(output_tensor)

    def print_sim_state(self):
        print('Fired:\n', self.Fired)
        print('Firing_idcs:\n', self.Firing_idcs)
        print('Firing_times:\n', self.Firing_times)
        print('Firing_counts:\n', self.Firing_counts)

    @property
    def type_groups(self) -> list[NeuronTypeGroup]:
        return list(self._type_group_dct.values())

    @property
    def type_group_conns(self) -> list[NeuronTypeGroupConnection]:
        return list(self._type_group_conn_dct.values())

    def _N_G_and_G_neuron_counts_1of2(self, shapes: NetworkArrayShapes, grid: NetworkGrid, neurons: Neurons):
        # N_G = self.izeros(shapes.N_G)
        # t_neurons_ids = torch.arange(self.N_G.shape[0], device='cuda')  # Neuron Id
        for g in self.type_groups:
            self.N_flags.type[g.start_idx:g.end_idx + 1] = g.ntype.value  # Set Neuron Type

        # rows[0, 1]: inhibitory count, excitatory count,
        # rows[2 * D]: number of neurons per delay (post_synaptic type: inhibitory, excitatory)
        G_neuron_counts = self.izeros(shapes.G_neuron_counts)
        snn_construction_gpu.fill_N_flags_group_id_and_G_neuron_count_per_type(
            N=self._config.N, G=self._config.G,
            N_pos=self.N_pos.data_ptr(),
            N_pos_shape=self._config.N_pos_shape,
            N_flags=self.N_flags.data_ptr(),
            G_shape=grid.segmentation,
            G_neuron_counts=G_neuron_counts.data_ptr(),
            N_flags_row_type=self.N_flags.rows.type.index,
            N_flags_row_group=self.N_flags.rows.group.index
        )

        G_neuron_typed_ccount = self.izeros((2 * self._config.G + 1))
        G_neuron_typed_ccount[1:] = G_neuron_counts[: 2, :].ravel().cumsum(dim=0)
        self.N_flags.validate(neurons, self.N_pos)
        return G_neuron_counts, G_neuron_typed_ccount

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

    def _G_delay_distance(self, G_pos: RegisteredVBO):
        # return None, None
        G_pos_distance = torch.cdist(G_pos.tensor, G_pos.tensor)
        return G_pos_distance, ((self._config.D - 1) * G_pos_distance / G_pos_distance.max()).round().int()

    def _curand_states(self):
        cu = snn_construction_gpu.CuRandStates(self._config.N).ptr()
        self.print_allocated_memory('curand_states')
        return cu

    @property
    def _N_pos_face_color(self):
        return self.N_pos.tensor[:, 7:11]

    @property
    def _N_pos_edge_color(self):
        return self.N_pos.tensor[:, 3:7]

    def _N_pos(self, shape, vbo):
        N_pos = RegisteredVBO(vbo, shape, self.device)
        for g in self.type_groups:
            if g.ntype == NeuronTypes.INHIBITORY:
                orange = torch.Tensor([1, .5, .2])
                N_pos.tensor[g.start_idx:g.end_idx + 1, 7:10] = orange  # Inhibitory Neurons -> Orange
        self.registered_buffers.append(N_pos)
        return N_pos

    # def _G_pos(self, shape, vbo) -> RegisteredVBO:
    #     # groups = torch.arange(self._config.G, device=self.device)
    #     # z = (groups / (self._config.G_shape[0] * self._config.G_shape[1])).floor()
    #     # r = groups - z * (self._config.G_shape[0] * self._config.G_shape[1])
    #     # y = (r / self._config.G_shape[0]).floor()
    #     # x = r - y * self._config.G_shape[0]
    #     #
    #     # gpos = torch.zeros(shape, dtype=torch.float32, device=self.device)
    #     #
    #     # gpos[:, 0] = x * (self._config.N_pos_shape[0] / self._config.G_shape[0])
    #     # gpos[:, 1] = y * (self._config.N_pos_shape[1] / self._config.G_shape[1])
    #     # gpos[:, 2] = z * (self._config.N_pos_shape[2] / self._config.G_shape[2])
    #
    #     G_pos = RegisteredVBO.from_buffer(
    #         vbo, config=GPUArrayConfig(shape=shape, strides=(shape[1] * 4, 4),
    #                                    dtype=np.float32, device=self.device))
    #
    #     # self.validate_N_G()
    #     return G_pos

    def set_src_group_weights(self, groups, w):
        selected = self.N_flags.select_by_groups(groups)
        self.synapse_arrays.N_weights[:, selected] = w

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

    def select_groups(self, mask):
        return self.G_flags.group_ids[mask]

    @staticmethod
    def actualize_group_separator_lines(plot_slots_tensor, pos_tensor, color_tensor, separator_mask, n_plots):

        separator_mask_ = separator_mask[: min(n_plots + 1, plot_slots_tensor[-1] + 1)].clone()
        separator_mask_[-1] = True
        separators = (plot_slots_tensor[: len(separator_mask_)][separator_mask_]
                      .repeat_interleave(2).to(torch.float32))

        separators = separators[: min(len(separators), pos_tensor.shape[0])]
        pos_tensor[:len(separators), 1] = separators
        color_tensor[:, 3] = 0
        color_tensor[:len(separators), 3] = 1

    def actualize_plot_map(self, groups):
        # selected_neurons = self.neuron_ids[self.selected_neuron_mask(groups)]
        selected_neurons = self.N_flags.select_ids_by_groups(groups)

        n_selected = len(selected_neurons)

        n_voltage_plots = min(n_selected, self._plotting_config.n_voltage_plots)
        self.plotting_arrays.voltage_map[: n_voltage_plots] = selected_neurons[: n_voltage_plots]

        n_scatter_plots = min(n_selected, self._plotting_config.n_scatter_plots)
        self.plotting_arrays.firings_map[: n_scatter_plots] = selected_neurons[: n_scatter_plots]

        if n_selected < self._plotting_config.n_voltage_plots:
            pass
        if n_selected < self._plotting_config.n_scatter_plots:
            pass

        neuron_groups = self.izeros(max(n_voltage_plots, n_scatter_plots) + 1)
        neuron_groups[: -1] = self.N_flags.group[selected_neurons[: max(n_voltage_plots, n_scatter_plots)]]
        neuron_groups_prev = self.izeros(neuron_groups.shape)
        neuron_groups_prev[0] = -1
        neuron_groups_prev[1:] = neuron_groups[0: -1]
        neuron_groups[-1] = -1

        separator_mask = neuron_groups != neuron_groups_prev

        self.actualize_group_separator_lines(
            plot_slots_tensor=self.plotting_arrays.voltage_plot_slots,
            separator_mask=separator_mask,
            pos_tensor=self.plotting_arrays.voltage_group_line_pos.tensor,
            color_tensor=self.plotting_arrays.voltage_group_line_colors.tensor,
            n_plots=n_voltage_plots)

        self.actualize_group_separator_lines(
            plot_slots_tensor=self.plotting_arrays.firings_plot_slots,
            separator_mask=separator_mask,
            pos_tensor=self.plotting_arrays.firings_group_line_pos.tensor,
            color_tensor=self.plotting_arrays.firings_group_line_colors.tensor,
            n_plots=n_scatter_plots)

    # def redirect_synapses(self, groups, direction, rate):
    #     pass

    @property
    def active_sensory_groups(self):
        return self.select_groups(self.G_flags.b_sensory_input.type(torch.bool))

    @property
    def active_output_groups(self):
        return self.select_groups(self.G_flags.b_output_group.type(torch.bool))

    def unregister_registered_buffers(self):
        for rb in self.registered_buffers:
            rb.reg.unregister(None)

    def update(self):

        if self.Simulation.t % 100 == 0:
            print('t =', self.Simulation.t)
        if self.Simulation.t % 1000 == 0:
            # if False:
            self.Simulation.calculate_avg_group_weight()
            a = self.to_dataframe(self.g2g_info_arrays.G_avg_weight_inh)
            b = self.to_dataframe(self.g2g_info_arrays.G_avg_weight_exc)
            r = 6
            # self.look_up([(80, 72), (88, 80), (96, 88), (104, 96), (112, 104)],
            # self.G_stdp_config0.type(torch.float32))
            # self.look_up([(80, 72), (88, 80), (96, 88), (104, 96), (112, 104)], self.G_avg_weight_inh)
            # self.look_up([(80, 72), (88, 80), (96, 88), (104, 96), (112, 104)], self.G_avg_weight_exc)
            # print()
            self.look_up([(75, 67), (83, 75), (91, 83), (99, 91), (107, 99), (115, 107), (123, 115)],
                         self.g2g_info_arrays.G_avg_weight_inh)
            self.look_up([(75, 67), (83, 75), (91, 83), (99, 91), (107, 99), (115, 107), (123, 115)],
                         self.g2g_info_arrays.G_avg_weight_exc)
            print()

        n_updates = self._config.sim_updates_per_frame

        t_mod = self.Simulation.t % self._plotting_config.scatter_plot_length

        if self._config.debug is False:

            for i in range(n_updates):
                # if self._config.debug is False:
                self.Simulation.update(self._config.stdp_active, self._config.debug)
                # if self.Simulation.t >= 2100:
                #     self.debug = True
        else:
            a = self.to_dataframe(self.Firing_idcs)
            b = self.to_dataframe(self.Firing_times)
            c = self.to_dataframe(self.Firing_counts)
            self.Simulation.update(self._config.stdp_active, False)

        # print(self.G_firing_count_hist.flatten()[67 + (self.Simulation.t-1) * self._config.G])

        self.plotting_arrays.group_firing_counts_plot_single1.tensor[
            t_mod: t_mod + n_updates, 1] = \
            self.G_firing_count_hist[t_mod: t_mod + n_updates, 123] / self.G_neuron_counts[1, 123]

        offset1 = self._plotting_config.scatter_plot_length

        self.plotting_arrays.group_firing_counts_plot_single1.tensor[
            offset1 + t_mod: offset1 + t_mod + n_updates, 1] = \
            self.G_firing_count_hist[t_mod: t_mod + n_updates, 125] / self.G_neuron_counts[1, 125]

        if t_mod + n_updates + 1 >= self._plotting_config.scatter_plot_length:
            self.G_firing_count_hist[:] = 0
