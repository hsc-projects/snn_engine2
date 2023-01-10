import numpy as np
import torch

from network.network_array_shapes import NetworkArrayShapes

from network.network_structures import NeuronTypes
from network.network_grid import NetworkGrid
from network.gpu.visualized_elements.synapse_visuals import VisualizedSynapsesCollection

# noinspection PyUnresolvedReferences
from network.gpu.cpp_cuda_backend import (
    snn_construction_gpu,
    snn_simulation_gpu
)
from network.gpu.neurons import NeuronRepresentation
from rendering import (
    GPUArrayCollection
)


# noinspection PyPep8Naming
class SynapseRepresentation(GPUArrayCollection):

    def __init__(self,
                 scene,
                 view,
                 neurons: NeuronRepresentation,
                 device: int,
                 shapes: NetworkArrayShapes,
                 ):

        self.neurons = neurons
        super().__init__(device=device, bprint_allocated_memory=self.neurons._config.N > 1000)

        self.pre_synaptic_rep_initialized = False

        self.N_delays = self.izeros(shapes.N_delays)
        self.N_rep = self.izeros(shapes.N_rep_inv)
        self.N_rep_buffer = self.izeros(shapes.N_rep)

        self.N_rep_pre_synaptic = self.izeros(shapes.N_rep_inv)
        self.N_rep_pre_synaptic_idcs = self.izeros(shapes.N_rep_inv)
        self.N_rep_pre_synaptic_counts = self.izeros(self.neurons._config.N + 1)

        self.N_weights = self.fzeros(shapes.N_weights)

        self.L_winner_take_all_map = self.izeros(shapes.L_winner_take_all_map) - 1
        self.L_winner_take_all_map[self.neurons._config.max_winner_take_all_layer_size, :] = 0
        self.n_wta_layers = 0

        self.RepBackend = snn_construction_gpu.SnnRepresentation(
            N=self.neurons._config.N,
            G=self.neurons._config.G,
            S=self.neurons._config.S,
            D=self.neurons._config.D,
            curand_states_p=self.neurons.curand_states,
            N_pos=self.neurons._neuron_visual.gpu_array.data_ptr(),
            G_group_delay_counts=self.neurons.G_group_delay_counts.data_ptr(),
            G_flags=self.neurons.G_flags.data_ptr(),
            G_props=self.neurons.G_props.data_ptr(),
            N_rep=self.N_rep.data_ptr(),
            N_rep_buffer=self.N_rep_buffer.data_ptr(),
            N_rep_pre_synaptic=self.N_rep_pre_synaptic.data_ptr(),
            N_rep_pre_synaptic_idcs=self.N_rep_pre_synaptic_idcs.data_ptr(),
            N_rep_pre_synaptic_counts=self.N_rep_pre_synaptic_counts.data_ptr(),
            N_delays=self.N_delays.data_ptr(),
            N_flags=self.neurons.N_flags.data_ptr(),
            N_weights=self.N_weights.data_ptr(),
            L_winner_take_all_map=self.L_winner_take_all_map.data_ptr(),
            max_n_winner_take_all_layers=self.neurons._config.max_n_winner_take_all_layers,
            max_winner_take_all_layer_size=self.neurons._config.max_winner_take_all_layer_size
        )

        (self.G_conn_probs,
         self.G_exp_ccsyn_per_src_type_and_delay,
         self.G_exp_exc_ccsyn_per_snk_type_and_delay) = self._fill_syn_counts(
            type_groups=self.neurons.type_groups,
            type_group_conns=self.neurons.type_group_conns,
            shapes=shapes,
            G_neuron_counts=self.neurons.G_neuron_counts)

        self.print_allocated_memory('N_rep_buffer')

        self.init_N_rep_and_N_delays(
            shapes=shapes,
            type_group_conns=self.neurons.type_group_conns)

        self.print_allocated_memory('N_rep_inv')

        self.N_rep_groups_cpu = self._N_rep_groups_cpu()

        self.init_N_weights(self.neurons.type_group_conns)
        self.print_allocated_memory('weights')

        self.G_swap_tensor = self._G_swap_tensor()
        self.N_relative_G_indices = self._N_relative_G_indices()
        self.print_allocated_memory('G_swap_tensor')

        self.visualized_synapses = VisualizedSynapsesCollection(
            scene=scene, view=view, device=device, neurons=self.neurons,
            synapses=self
        )

    def actualize_N_rep_pre_synaptic_idx(self, shapes):

        self.N_rep_buffer = self.N_rep_buffer.reshape(shapes.N_rep_inv)

        self.RepBackend.actualize_N_rep_pre_synaptic()

        if self.neurons._config.N <= 2 * 10 ** 5:

            a = self.to_dataframe(self.N_rep_buffer)
            b = self.to_dataframe(self.N_rep_pre_synaptic)
            c = self.to_dataframe(self.N_rep_pre_synaptic_idcs)
            d = self.to_dataframe(self.N_rep_pre_synaptic_counts)
            e = self.to_dataframe(self.N_rep)

            # noinspection PyTypeChecker
            assert len(self.N_rep_pre_synaptic_idcs[self.N_rep.flatten()[
                self.N_rep_pre_synaptic_idcs.type(torch.int64)] != self.N_rep_buffer]) == 0

        self.N_rep_buffer[:] = -1

        self.pre_synaptic_rep_initialized = True
    # noinspection PyUnusedLocal

    def _fill_syn_counts(self, type_groups, type_group_conns, shapes, G_neuron_counts):

        S, D, G = self.neurons._config.S, self.neurons._config.D, self.neurons._config.G

        G_conn_probs = self.fzeros(shapes.G_conn_probs)
        G_exp_ccsyn_per_src_type_and_delay = self.izeros(shapes.G_exp_ccsyn_per_src_type_and_delay)
        G_exp_exc_ccsyn_per_snk_type_and_delay = self.izeros(shapes.G_exp_exc_ccsyn_per_snk_type_and_delay)

        snn_construction_gpu.fill_G_exp_ccsyn_per_src_type_and_delay(
            S=S, D=D, G=G,
            G_neuron_counts=G_neuron_counts.data_ptr(),
            G_conn_probs=G_conn_probs.data_ptr(),
            G_exp_ccsyn_per_src_type_and_delay=G_exp_ccsyn_per_src_type_and_delay.data_ptr())

        exp_result = (self.fzeros(G) + 1) * S

        for ntype_group in type_groups:
            first_row = G_exp_ccsyn_per_src_type_and_delay[(D + 1) * (ntype_group.ntype - 1), :]
            if first_row.sum() != 0:
                print(first_row)
                raise AssertionError
            last_row = G_exp_ccsyn_per_src_type_and_delay[(D + 1) * (ntype_group.ntype - 1) + D, :]
            if ((last_row - exp_result).abs()).sum() != 0:
                print(last_row)
                print((last_row - exp_result).abs())
                raise AssertionError

        exc_syn_counts = []

        for gc in type_group_conns:
            if gc.src_type_value == NeuronTypes.EXCITATORY.value:
                exc_syn_counts.append(len(gc))
        assert np.array(exc_syn_counts).cumsum()[-1] == S

        max_median_inh_targets_delay = -1
        # max_inh_target_row = G_neuron_counts[2, :]
        max_median_exc_targets_delay = -1
        # max_exc_target_row = G_neuron_counts[2 + D, :]

        last_row_inh = None
        last_row_exc = None

        autapse_mask = torch.zeros(G, dtype=torch.bool, device=self.device)
        exp_inh = self.izeros(G)
        exp_exc = self.izeros(G)
        row_exc_max = D + 2

        def add_mask(row, v, mask=None):
            if mask is not None:
                G_exp_exc_ccsyn_per_snk_type_and_delay[row, :][mask] = (
                        G_exp_exc_ccsyn_per_snk_type_and_delay[row, :] + v)[mask]
            else:
                G_exp_exc_ccsyn_per_snk_type_and_delay[row, :] += v

        def row_diff(row):
            return (G_exp_exc_ccsyn_per_snk_type_and_delay[row, :]
                    - G_exp_exc_ccsyn_per_snk_type_and_delay[row - 1, :])

        def inh_targets(delay):
            return G_neuron_counts[2 + delay, :]

        def exc_targets(delay):
            return G_neuron_counts[2 + D + delay, :]

        for d in range(D):

            row_inh = d + 1
            row_exc = D + 2 + d

            if d > 0:
                if max_median_inh_targets_delay < inh_targets(d).median():
                    max_median_inh_targets_delay = d
                    # max_inh_target_row = inh_targets(d)
                if max_median_exc_targets_delay < exc_targets(d).median():
                    max_median_exc_targets_delay = d
                    # max_exc_target_row = exc_targets(d)
                    row_exc_max = row_exc

            exc_ccsyn = G_exp_ccsyn_per_src_type_and_delay[row_exc, :]
            exp_inh[:] = exc_ccsyn * (exc_syn_counts[0]/S) + .5
            add_mask(row_inh, exp_inh)
            exp_exc[:] = exc_ccsyn * (exc_syn_counts[1]/S) + .5
            add_mask(row_exc, exp_exc)

            if d == 0:
                autapse_mask[:] = (exp_exc == exc_targets(d)) & (exp_exc > 0)
            add_mask(row_exc, -1, autapse_mask)

            exp_inh_count = row_diff(row_inh)
            exp_exc_count = row_diff(row_exc)

            inh_count_too_high_mask = (inh_targets(d) - exp_inh_count) < 0
            if inh_count_too_high_mask.any():
                if (d > 0) & ((row_diff(row_inh-1) < inh_targets(d-1)) & inh_count_too_high_mask).all():
                    add_mask(row_inh - 1, 1, mask=inh_count_too_high_mask)
                else:
                    add_mask(row_inh, - 1, mask=inh_count_too_high_mask)
                    # add_mask(row_inh + 1, 1, mask=inh_count_too_high_mask & (row_diff(row_inh + 1) < 0))

            exc_count_too_high_mask = (exc_targets(d) - exp_exc_count) < 0
            if exc_count_too_high_mask.any():
                if (d > 1) & ((row_diff(row_exc - 1) < exc_targets(d-1)) & exc_count_too_high_mask).all():
                    add_mask(row_exc - 1, 1, mask=exc_count_too_high_mask)
                else:
                    add_mask(row_exc, - 1, mask=exc_count_too_high_mask)
                    # add_mask(row_exc + 1, 1, mask=exc_count_too_high_mask)

            if d == (D-1):
                if (max_median_exc_targets_delay == 0) or (row_exc_max == D + 2):
                    raise AssertionError
        # print(G_exp_exc_ccsyn_per_snk_type_and_delay)
        for d in range(max_median_exc_targets_delay, D):
            row_inh = d + 1
            row_exc = D + 2 + d
            add_mask(row_exc, 1, mask=autapse_mask)

            exp_inh_count = row_diff(row_inh)
            exp_exc_count = row_diff(row_exc)

            if (exp_inh_count < 0).any():
                raise ValueError(f'({row_exc}){(exp_inh_count < 0).sum()}')
            if (exp_exc_count < 0).any():
                raise ValueError(f'({row_exc}){(exp_exc_count < 0).sum()}')

            if ((inh_targets(d) - exp_inh_count) < 0).any():
                raise ValueError(f'({row_exc}){((inh_targets(d) - exp_inh_count) < 0).sum()}')
            if ((exc_targets(d) - exp_exc_count) < 0).any():
                raise ValueError(f'({row_exc}){((exc_targets(d) - exp_exc_count) < 0).sum()}')

            if d == (D-1):
                if (max_median_exc_targets_delay == 0) or (row_exc_max == D + 2):
                    raise AssertionError
                last_row_inh = G_exp_exc_ccsyn_per_snk_type_and_delay[row_inh, :]
                last_row_exc = G_exp_exc_ccsyn_per_snk_type_and_delay[row_exc, :]

        # print(G_exp_exc_ccsyn_per_snk_type_and_delay)

        inh_neq_exp_mask = last_row_inh != exc_syn_counts[0]
        exc_neq_exp_mask = last_row_exc != exc_syn_counts[1]

        if any(inh_neq_exp_mask) or any(exc_neq_exp_mask):
            print('G_neuron_counts:\n', G_neuron_counts, '\n')
            print('G_exp_ccsyn_per_src_type_and_delay:\n', G_exp_ccsyn_per_src_type_and_delay, '\n')
            print('G_exp_exc_ccsyn_per_snk_type_and_delay:\n', G_exp_exc_ccsyn_per_snk_type_and_delay)
            raise AssertionError

        return G_conn_probs, G_exp_ccsyn_per_src_type_and_delay, G_exp_exc_ccsyn_per_snk_type_and_delay

    def _G_swap_tensor(self):
        max_neurons_per_group = self.group_counts(self.neurons.G_neuron_counts).max().item()
        m = self.neurons._config.swap_tensor_shape_multiplicators
        return self.izeros((m[0], m[1] * max_neurons_per_group)) - 1

    def init_N_rep_and_N_delays(
            self, type_group_conns,
            shapes: NetworkArrayShapes):

        N, S, D, G = self.neurons._config.N, self.neurons._config.S, self.neurons._config.D, self.neurons._config.G

        torch.cuda.empty_cache()
        self.print_allocated_memory('syn_counts')

        # N_delays = self.izeros(shapes.N_delays)
        # N_rep_t = self.izeros(shapes.N_rep_inv)
        torch.cuda.empty_cache()
        self.print_allocated_memory('N_rep')
        self.N_rep_buffer[:] = 0

        def cc_syn_(gc_):
            t = self.izeros((D + 1, G))
            if (gc_.src.ntype == NeuronTypes.INHIBITORY) and (gc_.snk.ntype == NeuronTypes.EXCITATORY):
                t[:, :] = self.G_exp_ccsyn_per_src_type_and_delay[0: D + 1, :]
            elif (gc_.src.ntype == NeuronTypes.EXCITATORY) and (gc_.snk.ntype == NeuronTypes.INHIBITORY):
                t[:, :] = self.G_exp_exc_ccsyn_per_snk_type_and_delay[0: D + 1, :]
            elif (gc_.src.ntype == NeuronTypes.EXCITATORY) and (gc_.snk.ntype == NeuronTypes.EXCITATORY):
                t[:, :] = self.G_exp_exc_ccsyn_per_snk_type_and_delay[D + 1: 2 * (D + 1), :]
            else:
                raise ValueError
            return t

        for i, gc in enumerate(type_group_conns):
            # cn_row = gc.src_type_value - 1
            ct_row = (gc.snk_type_value - 1) * D + 2
            # slice_ = self.G_neuron_counts[cn_row, :]
            # counts = torch.repeat_interleave(self.G_neuron_counts[ct_row: ct_row+D, :].T, slice_, dim=0)

            ccn_idx_src = G * (gc.src_type_value - 1)
            ccn_idx_snk = G * (gc.snk_type_value - 1)

            G_autapse_indices = self.izeros((D, G))
            G_relative_autapse_indices = self.izeros((D, G))
            cc_syn = cc_syn_(gc)

            self.RepBackend.fill_N_rep(
                cc_src=self.neurons.G_neuron_typed_ccount[ccn_idx_src: ccn_idx_src + G + 1].data_ptr(),
                cc_snk=self.neurons.G_neuron_typed_ccount[ccn_idx_snk: ccn_idx_snk + G + 1].data_ptr(),
                G_rep=self.neurons.g2g_info_arrays.G_rep.data_ptr(),
                G_neuron_counts=self.neurons.G_neuron_counts[ct_row: ct_row+D, :].data_ptr(),
                G_autapse_indices=G_autapse_indices.data_ptr(),
                G_relative_autapse_indices=G_relative_autapse_indices.data_ptr(),
                has_autapses=ccn_idx_src == ccn_idx_snk,
                gc_location=gc.location,
                gc_conn_shape=gc.conn_shape,
                cc_syn=cc_syn.data_ptr(),
                sort_keys=self.N_rep_buffer.data_ptr(),
                verbose=False)

            if G_autapse_indices[1:, :].sum() != -(G_autapse_indices.shape[0] - 1) * G_autapse_indices.shape[1]:
                raise AssertionError

            if (G_relative_autapse_indices[1:, :].sum()
                    != -(G_relative_autapse_indices.shape[0] - 1) * G_relative_autapse_indices.shape[1]):
                raise AssertionError

        del G_autapse_indices
        del G_relative_autapse_indices
        torch.cuda.empty_cache()

        snn_construction_gpu.sort_N_rep(N=N, S=S, sort_keys=self.N_rep_buffer.data_ptr(),
                                        N_rep=self.N_rep.data_ptr())

        for i, gc in enumerate(type_group_conns):

            ct_row = (gc.snk_type_value - 1) * D + 2

            ccn_idx_src = G * (gc.src_type_value - 1)
            ccn_idx_snk = G * (gc.snk_type_value - 1)
            cc_syn = cc_syn_(gc)

            snn_construction_gpu.reindex_N_rep(
                N=N, S=S, D=D, G=G,
                N_flags=self.neurons.N_flags.data_ptr(),
                cc_src=self.neurons.G_neuron_typed_ccount[ccn_idx_src: ccn_idx_src + G + 1].data_ptr(),
                cc_snk=self.neurons.G_neuron_typed_ccount[ccn_idx_snk: ccn_idx_snk + G + 1].data_ptr(),
                G_rep=self.neurons.g2g_info_arrays.G_rep.data_ptr(),
                G_neuron_counts=self.neurons.G_neuron_counts[ct_row: ct_row+D, :].data_ptr(),
                G_group_delay_counts=self.neurons.G_group_delay_counts.data_ptr(),
                gc_location=gc.location,
                gc_conn_shape=gc.conn_shape,
                cc_syn=cc_syn.data_ptr(),
                N_delays=self.N_delays.data_ptr(),
                sort_keys=self.N_rep_buffer.data_ptr(),
                N_rep=self.N_rep.data_ptr(),
                N_flags_row_group=self.neurons.N_flags.rows.group.index,
                verbose=False)

        snn_construction_gpu.sort_N_rep(N=N, S=S, sort_keys=self.N_rep_buffer.data_ptr(),
                                        N_rep=self.N_rep.data_ptr())

        # N_rep = torch.empty(shapes.N_rep, dtype=torch.int32, device=self.device)
        self.N_rep_buffer = self.N_rep_buffer.reshape(shapes.N_rep)
        self.N_rep_buffer[:] = -1
        self.N_rep_buffer[:] = self.N_rep.T

        self.N_rep = self.N_rep.reshape(shapes.N_rep)
        self.N_rep[:] = self.N_rep_buffer
        self.N_rep_buffer[:] = -1
        self.print_allocated_memory(f'transposed')

        assert len(self.N_rep[self.N_rep == -1]) == 0

    def _N_rep_groups_cpu(self):
        N_rep_groups = self.N_rep.clone()
        # for ntype in NeuronTypes:
        #     for g in range(self._config.G):
        #         col = 2 * (ntype - 1)
        #         N_rep_groups[((self.N_rep >= self.group_indices[g, col])
        #                      & (self.N_rep <= self.group_indices[g, col + 1]))] = g

        snn_construction_gpu.fill_N_rep_groups(
            N=self.neurons._config.N,
            S=self.neurons._config.S,
            N_flags=self.neurons.N_flags.data_ptr(),
            N_rep=self.N_rep.data_ptr(),
            N_rep_groups=N_rep_groups.data_ptr(),
            N_flags_row_group=self.neurons.N_flags.rows.group.index,
        )
        self.print_allocated_memory('N_rep_groups')
        return N_rep_groups.cpu()

    def init_N_weights(self, type_group_conns):
        for gc in type_group_conns:
            if not isinstance(gc.w0, str):
                self.N_weights[gc.location[1]: gc.location[1] + gc.conn_shape[1],
                               gc.location[0]: gc.location[0] + gc.conn_shape[0]] = gc.w0

            elif (gc.w0 == 'r') or (gc.w0 == '-r'):
                r = torch.rand((gc.conn_shape[1], gc.conn_shape[0]))
                if gc.w0 == '-r':
                    r = -r
                self.N_weights[gc.location[1]: gc.location[1] + gc.conn_shape[1],
                               gc.location[0]: gc.location[0] + gc.conn_shape[0]] = r
            else:
                raise NotImplementedError

    def _N_relative_G_indices(self):
        all_groups = self.neurons.N_flags.group.type(torch.int64)
        inh_start_indices = self.neurons.G_neuron_typed_ccount[all_groups]
        start_indices = self.neurons.G_neuron_typed_ccount[all_groups + self.neurons._config.G]
        inh_neurons = self.neurons.N_flags.type == NeuronTypes.INHIBITORY.value
        start_indices[inh_neurons] = inh_start_indices[inh_neurons]
        start_indices[~inh_neurons] -= (self.neurons.G_neuron_typed_ccount[all_groups + 1][~inh_neurons]
                                        - inh_start_indices[~inh_neurons])
        # return (self.neuron_ids - start_indices).type(torch.int32)
        return (self.neurons.N_flags.id - start_indices).type(torch.int32)

    def _single_group_swap(self,
                           program,
                           group_neuron_counts_total, group_neuron_counts_typed,
                           neuron_group_counts, neuron_group_indices,
                           swap_rates):

        # swap_delay0 = self.G_delay_distance[program[0, 0], program[1, 0]].item()

        neurons = self.neurons.N_flags.select_ids_by_groups(groups=program[1]).type(torch.int64)
        n_neurons = len(neurons)

        assert (neuron_group_indices[neuron_group_indices >= 0]
                + program[1, 0] - self.neurons.N_flags.group[neurons]).sum() == 0

        print_idx = min(1900000, n_neurons - 1)
        # print_idx = 0

        assert neurons.dtype == torch.int64
        assert program.dtype == torch.int64
        self.RepBackend.swap_groups(
            neurons.data_ptr(), n_neurons,
            program.data_ptr(), program.shape[1],
            neuron_group_indices.data_ptr(),
            self.G_swap_tensor.data_ptr(), self.G_swap_tensor.shape[1],
            swap_rates.data_ptr(), swap_rates.data_ptr(),
            group_neuron_counts_typed[0].data_ptr(), group_neuron_counts_typed[1].data_ptr(),
            group_neuron_counts_total.data_ptr(),
            self.neurons.g2g_info_arrays.G_delay_distance.data_ptr(),
            self.N_relative_G_indices.data_ptr(),
            self.neurons.G_neuron_typed_ccount.data_ptr(), neuron_group_counts.data_ptr(),
            print_idx)
        # noinspection PyUnusedLocal
        a, b = self.swap_validation(print_idx, neurons)
        # assert (neuron_group_counts[0].any() == False)
        self.N_rep_groups_cpu[:, neurons] = self.G_swap_tensor[:, :n_neurons].cpu()

        self.G_swap_tensor[:] = -1

        neuron_group_counts[:] = 0

    def add_winner_take_all_layer(self, neurons):
        assert self.n_wta_layers < self.neurons._config.max_n_winner_take_all_layers
        assert len(neurons) <= self.neurons._config.max_winner_take_all_layer_size
        if isinstance(neurons, list):
            neurons = torch.Tensor([neurons]).to(self.device)

        self.L_winner_take_all_map[:len(neurons), self.n_wta_layers] = neurons
        self.L_winner_take_all_map[self.neurons._config.max_winner_take_all_layer_size,
                                   self.n_wta_layers] = len(neurons)

    @staticmethod
    def group_counts(G_neuron_counts):
        return G_neuron_counts[:len(NeuronTypes)].sum(axis=0)

    # noinspection PyUnusedLocal
    def make_sensory_groups(
            self,
            G_neuron_counts,
            N_pos, G_pos,
            groups, grid: NetworkGrid,
            single_neuron_input: bool = False
    ):
        """
        (Work in Progress)
        1. Make a sufficient number of sensory neurons: (Izhikevich-Model - tonic Bursting)
        2. Connect each sensory neurons to two unique filter neurons.
        3. Connect the filter neurons to winner takes all neurons.
        4. Change the neuron positions for visibility
        """

        for group in groups:
            ntype = NeuronTypes.EXCITATORY.value

            neuron_group_ids_exc = self.neurons.N_flags.select_ids_by_groups([group], ntype=NeuronTypes.EXCITATORY.value)

            if single_neuron_input is True:
                n_sensory_neurons = max(int(G_neuron_counts[ntype - 1, :][group]/self.neurons._config.S), 1)
                sensory_neuron_ids = neuron_group_ids_exc[:n_sensory_neurons]
            else:
                n_sensory_neurons = None
                sensory_neuron_ids = neuron_group_ids_exc

            self.neurons.N_flags.b_sensory_input[sensory_neuron_ids] = 1

            if single_neuron_input is True:
                n_winner_take_all_neurons = 2
                filter_neuron_ids_inh = self.neurons.N_flags.select_ids_by_groups([group], ntype=NeuronTypes.INHIBITORY.value)
                filter_neuron_ids_exc = neuron_group_ids_exc[n_sensory_neurons:-n_winner_take_all_neurons]

                ux = grid.unit_shape[0]
                if n_sensory_neurons > 1:
                    new_x_coords = torch.linspace(G_pos.tensor[group][0] + ux * 0.1,
                                                  G_pos.tensor[group][0] + ux * 0.9,
                                                  steps=n_sensory_neurons, device=self.device)
                else:
                    new_x_coords = G_pos.tensor[group][0] + ux * 0.5
                new_y_coords = G_pos.tensor[group][1]
                new_z_coords = G_pos.tensor[group][2] + (grid.unit_shape[2] / 2)
                N_pos.tensor[sensory_neuron_ids, 0] = new_x_coords
                N_pos.tensor[sensory_neuron_ids, 1] = new_y_coords
                N_pos.tensor[sensory_neuron_ids, 2] = new_z_coords
                N_pos.tensor[sensory_neuron_ids, 10] = 1

                uy = grid.unit_shape[1]
                max_y = G_pos.tensor[group][1] + uy
                # self.N_pos.tensor[filter_neuron_ids_exc, 10] = 1
                N_pos.tensor[filter_neuron_ids_exc, 1] += .33 * uy * (
                        (max_y - N_pos.tensor[filter_neuron_ids_exc, 1]) / max_y)
                N_pos.tensor[filter_neuron_ids_inh, 1] += .33 * uy * (
                        (max_y - N_pos.tensor[filter_neuron_ids_inh, 1]) / max_y)

                # print(self.N_rep[:, sensory_neuron_ids[0]])

            self.RepBackend.nullify_all_weights_to_group(group)
            # print(self.N_rep[:, sensory_neuron_ids[0]])
            return

        return

    def swap_group_synapses(self,
                            groups, self_swap=True):

        # groups = groups[:3:, [0]]
        # groups = groups.flip(0)

        n_groups = groups.shape[1]
        if n_groups > self.neurons._config.swap_tensor_shape_multiplicators[1]:
            raise ValueError

        swap_delay = self.neurons.g2g_info_arrays.G_delay_distance[groups.ravel()[0], groups.ravel()[n_groups]].item()
        # noinspection PyUnresolvedReferences
        if not bool((self.neurons.g2g_info_arrays.G_delay_distance[
                         groups.ravel()[:groups.size().numel() - n_groups],
                         groups.ravel()[n_groups:groups.size().numel()]] == swap_delay).all()):
            raise ValueError

        chain_length = groups.shape[0] - 2

        group_neuron_counts_typed = (self.neurons.G_neuron_counts[:len(NeuronTypes)][:, groups.ravel()]
                                     .reshape((2, groups.shape[0], groups.shape[1])))

        group_neuron_counts_total = group_neuron_counts_typed[0] + group_neuron_counts_typed[1]

        inh_sums = group_neuron_counts_typed[0].sum(axis=1)
        total_sums = group_neuron_counts_total.sum(axis=1)

        swap_rates0 = self.fzeros((chain_length, n_groups)) + 1.
        swap_rates1 = self.fzeros((chain_length, n_groups)) + 1.
        # group_indices_offset = 0
        # max_neurons_per_group = int(self.G_swap_tensor.shape[0] / 2)
        neuron_group_indices = self.izeros(self.G_swap_tensor.shape[1]) - 1
        neuron_group_counts = self.izeros((2, self.G_swap_tensor.shape[1]))
        neuron_group_indices_aranged = torch.arange(n_groups, device=self.device)

        for i in range(chain_length):
            program = groups[i:i + 3]

            n_inh_core_neurons = inh_sums[i + 1]
            neuron_group_indices[:n_inh_core_neurons] = (
                torch.repeat_interleave(neuron_group_indices_aranged, group_neuron_counts_typed[0][i + 1].ravel()))
            neuron_group_indices[n_inh_core_neurons:total_sums[i+1]] = (
                torch.repeat_interleave(neuron_group_indices_aranged, group_neuron_counts_typed[1][i + 1].ravel()))

            self._single_group_swap(
                program=program,
                group_neuron_counts_total=group_neuron_counts_total[i:i + 3],
                group_neuron_counts_typed=group_neuron_counts_typed[:, i:i + 3],
                neuron_group_counts=neuron_group_counts,
                neuron_group_indices=neuron_group_indices,
                swap_rates=swap_rates0[i])

            self._single_group_swap(
                program=program[[1, 1, 2], :].clone(),
                group_neuron_counts_total=group_neuron_counts_total[[i+1, i+1, i+2], :].clone(),
                group_neuron_counts_typed=group_neuron_counts_typed[:, [i+1, i+1, i+2]].clone(),
                neuron_group_counts=neuron_group_counts,
                neuron_group_indices=neuron_group_indices,
                swap_rates=swap_rates1[i])

            neuron_group_indices[:] = -1

        assert len(self.N_rep[self.N_rep < 0]) == 0
        return

    def set_weight(self, w, x0=0, x1=None, y0=0, y1=None):
        self.N_weights[x0: x1 if x1 is not None else self.neurons._config.N,
                       y0: y1 if y1 is not None else self.neurons._config.S] = w

    def swap_validation(self, j, neurons):
        a = self.to_dataframe(self.G_swap_tensor)
        b = a.iloc[:, j:j + 3].copy()
        b.columns = [1, 2, 3]
        b[2] = self.N_rep[:, neurons[j]].cpu().numpy()
        b[3] = self.neurons.N_flags.group[b[2]].cpu().numpy()
        b[0] = self.N_rep_groups_cpu[:, neurons[j]].numpy()
        b[4] = b[0] != b[3]

        return a, b

    def unregister_registered_buffers(self):
        super().unregister_registered_buffers()
        self.visualized_synapses.unregister_registered_buffers()