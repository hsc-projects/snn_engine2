import numpy as np
import torch
from typing import Union

# from network.gpu.plotting import PlottingGPUArrays
from network.network_config import NetworkConfig, PlottingConfig
from network.gpu.visualized_elements import VoltageMultiPlot, FiringScatterPlot, GroupFiringCountsPlot
from network.gpu.visualized_elements.boxes import GroupInfo

# noinspection PyUnresolvedReferences
from network.gpu.cpp_cuda_backend import (
    snn_construction_gpu,
    snn_simulation_gpu
)
from network.gpu.neurons import NeuronRepresentation
from network.gpu.synapses import SynapseRepresentation
# from network.gpu.chemicals import ChemicalRepresentation
from network.chemical_config import ChemicalConfigCollection, DefaultChemicals
from rendering import GPUArrayCollection
# from network.spiking_neural_network import SpikingNeuralNetwork


# noinspection PyPep8Naming
class NetworkSimulationGPU(GPUArrayCollection):

    def __init__(self,
                 engine,
                 config: NetworkConfig,
                 device: int,
                 T: int,
                 neurons: NeuronRepresentation,
                 synapses: SynapseRepresentation,
                 plotting_config: PlottingConfig,
                 group_firing_counts_plot_single1: GroupFiringCountsPlot,
                 chemical_concentrations: Union[ChemicalConfigCollection, DefaultChemicals]
                 ):
        super().__init__(device=device, bprint_allocated_memory=config.N > 1000)

        self._config: NetworkConfig = config
        self._plotting_config: PlottingConfig = plotting_config

        if group_firing_counts_plot_single1 is not None:
            self._group_firing_counts_plot_single1_tensor = group_firing_counts_plot_single1.vbo_array.tensor

        self.neurons = neurons
        self.synapse_arrays = synapses

        if plotting_config.has_voltage_multiplot:
            self._voltage_multiplot = VoltageMultiPlot(
                scene=engine.voltage_multiplot_scene,
                view=engine.voltage_multiplot_view,
                device=device,
                n_plots=self._plotting_config.n_voltage_plots,
                plot_length=self._plotting_config.voltage_plot_length,
                n_group_separator_lines=self.neurons.data_shapes.n_group_separator_lines
            )
            self.registered_buffers += self._voltage_multiplot.registered_buffers
        else:
            self._voltage_multiplot = None

        if plotting_config.has_firing_scatterplot:
            self._firing_scatter_plot = FiringScatterPlot(
                scene=engine.firing_scatter_plot_scene,
                view=engine.firing_scatter_plot_view,
                device=device,
                n_plots=self._plotting_config.n_scatter_plots,
                plot_length=self._plotting_config.scatter_plot_length,
                n_group_separator_lines=self.neurons.data_shapes.n_group_separator_lines
            )
            self.registered_buffers += self._firing_scatter_plot.registered_buffers
        else:
            self._firing_scatter_plot = None

        self.Fired = self.fzeros(self._config.N)
        self.last_Fired = self.izeros(self._config.N) - self._config.D
        self.Firing_times = self.fzeros(self.neurons.data_shapes.Firing_times)
        self.Firing_idcs = self.izeros(self.neurons.data_shapes.Firing_idcs)
        self.Firing_counts = self.izeros(self.neurons.data_shapes.Firing_counts)

        self.G_firing_count_hist = self.izeros((self._plotting_config.scatter_plot_length, self._config.G))

        self.lookup_output_tensor = self.fzeros(6)

        self.chemical_concentrations = chemical_concentrations

        self.Simulation = snn_simulation_gpu.SnnSimulation(
            N=self._config.N,
            G=self._config.G,
            S=self._config.S,
            D=self._config.D,
            T=T,
            n_voltage_plots=plotting_config.n_voltage_plots,
            voltage_plot_length=plotting_config.voltage_plot_length,
            voltage_plot_data=self._voltage_multiplot.vbo_array.data_ptr(),
            voltage_plot_map=self._voltage_multiplot.map.data_ptr(),
            n_scatter_plots=plotting_config.n_scatter_plots,
            scatter_plot_length=plotting_config.scatter_plot_length,
            scatter_plot_data=self._firing_scatter_plot.vbo_array.data_ptr(),
            scatter_plot_map=self._firing_scatter_plot.map.data_ptr(),
            curand_states_p=self.neurons.curand_states,
            N_pos=self.neurons._neuron_visual.gpu_array.data_ptr(),
            # N_G=self.N_G.data_ptr(),
            G_group_delay_counts=self.neurons.G_group_delay_counts.data_ptr(),
            G_flags=self.neurons.G_flags.data_ptr(),
            G_props=self.neurons.G_props.data_ptr(),
            N_rep=self.synapse_arrays.N_rep.data_ptr(),
            N_rep_buffer=self.synapse_arrays.N_rep_buffer.data_ptr(),
            N_rep_pre_synaptic=self.synapse_arrays.N_rep_pre_synaptic.data_ptr(),
            N_rep_pre_synaptic_idcs=self.synapse_arrays.N_rep_pre_synaptic_idcs.data_ptr(),
            N_rep_pre_synaptic_counts=self.synapse_arrays.N_rep_pre_synaptic_counts.data_ptr(),
            N_delays=self.synapse_arrays.N_delays.data_ptr(),
            N_flags=self.neurons.N_flags.data_ptr(),
            N_states=self.neurons.N_states.data_ptr(),
            N_weights=self.synapse_arrays.N_weights.data_ptr(),
            fired=self.Fired.data_ptr(),
            last_fired=self.last_Fired.data_ptr(),
            firing_times=self.Firing_times.data_ptr(),
            firing_idcs=self.Firing_idcs.data_ptr(),
            firing_counts=self.Firing_counts.data_ptr(),
            G_firing_count_hist=self.G_firing_count_hist.data_ptr(),
            G_stdp_config0=self.neurons.g2g_info_arrays.G_stdp_config0.data_ptr(),
            G_stdp_config1=self.neurons.g2g_info_arrays.G_stdp_config1.data_ptr(),
            G_avg_weight_inh=self.neurons.g2g_info_arrays.G_avg_weight_inh.data_ptr(),
            G_avg_weight_exc=self.neurons.g2g_info_arrays.G_avg_weight_exc.data_ptr(),
            G_syn_count_inh=self.neurons.g2g_info_arrays.G_syn_count_inh.data_ptr(),
            G_syn_count_exc=self.neurons.g2g_info_arrays.G_syn_count_exc.data_ptr(),
            L_winner_take_all_map=self.synapse_arrays.L_winner_take_all_map.data_ptr(),
            max_n_winner_take_all_layers=self._config.max_n_winner_take_all_layers,
            max_winner_take_all_layer_size=self._config.max_winner_take_all_layer_size,
            C_old=self.chemical_concentrations.C_old.data_ptr(),
            C_new=self.chemical_concentrations.C_new.data_ptr(),
            C_source=self.chemical_concentrations.C_source.data_ptr(),
            chem_grid_w=self.chemical_concentrations.width,
            chem_grid_h=self.chemical_concentrations.height,
            chem_grid_d=self.chemical_concentrations.depth,
            chem_k_val=self.chemical_concentrations.k_val,
            chem_depreciation=self.chemical_concentrations.depreciation
        )

        self.group_info_mesh = GroupInfo(
            scene=engine.group_info_scene,
            view=engine.group_info_view,
            network_config=self._config, grid=self._config.grid,
            connect=np.zeros((self._config.G + 1, 2)) + self._config.G,
            device=device, G_flags=self.neurons.G_flags, G_props=self.neurons.G_props,
            g2g_info_arrays=self.neurons.g2g_info_arrays
        )

        self.registered_buffers += self.group_info_mesh.registered_buffers

        engine.set_main_context_as_current()

    @classmethod
    def from_snn(cls, network, engine, device):
        return cls(engine=engine,
                   config=network.network_config,
                   device=device,
                   T=network.T,
                   neurons=network.neurons,
                   synapses=network.synapse_arrays,
                   plotting_config=network.plotting_config,
                   group_firing_counts_plot_single1=network.group_firing_counts_plot_single1,
                   chemical_concentrations=network.chemical_concentrations)

    def _post_synapse_mod_init(self, shapes=None):
        if shapes is None:
            shapes = self.neurons.data_shapes
        self.synapse_arrays.actualize_N_rep_pre_synaptic_idx(shapes=shapes)
        self.synapse_arrays.N_rep_buffer = self.synapse_arrays.N_rep_buffer.reshape(shapes.N_rep)
        self.Simulation.calculate_avg_group_weight()

    # noinspection PyUnusedLocal
    def look_up(self, tuples, input_tensor, output_tensor=None, precision=6):
        if output_tensor is None:
            if len(self.lookup_output_tensor) != len(tuples):
                self.lookup_output_tensor = self.fzeros(len(tuples))
            output_tensor = self.lookup_output_tensor
        output_tensor[:] = torch.nan
        for i, e in enumerate(tuples):
            output_tensor[i] = input_tensor[e]
        print(output_tensor)

    def print_sim_state(self):
        print('Fired:\n', self.Fired)
        print('Firing_idcs:\n', self.Firing_idcs)
        print('Firing_times:\n', self.Firing_times)
        print('Firing_counts:\n', self.Firing_counts)

    def actualize_plot_map(self, groups):
        # selected_neurons = self.neuron_ids[self.selected_neuron_mask(groups)]
        selected_neurons = self.neurons.N_flags.select_ids_by_groups(groups)

        n_selected = len(selected_neurons)

        n_voltage_plots = min(n_selected, self._plotting_config.n_voltage_plots)
        self._voltage_multiplot.map[: n_voltage_plots] = selected_neurons[: n_voltage_plots]

        n_scatter_plots = min(n_selected, self._plotting_config.n_scatter_plots)
        self._firing_scatter_plot.map[: n_scatter_plots] = selected_neurons[: n_scatter_plots]

        if n_selected < self._plotting_config.n_voltage_plots:
            pass
        if n_selected < self._plotting_config.n_scatter_plots:
            pass

        neuron_groups = self.izeros(max(n_voltage_plots, n_scatter_plots) + 1)
        neuron_groups[: -1] = self.neurons.N_flags.group[selected_neurons[: max(n_voltage_plots, n_scatter_plots)]]
        neuron_groups_prev = self.izeros(neuron_groups.shape)
        neuron_groups_prev[0] = -1
        neuron_groups_prev[1:] = neuron_groups[0: -1]
        neuron_groups[-1] = -1

        separator_mask = neuron_groups != neuron_groups_prev

        self._voltage_multiplot.actualize_group_separator_lines(separator_mask, n_voltage_plots)
        self._firing_scatter_plot.actualize_group_separator_lines(separator_mask, n_voltage_plots)

    def set_src_group_weights(self, groups, w):
        selected = self.neurons.N_flags.select_by_groups(groups)
        self.synapse_arrays.N_weights[:, selected] = w

    def update(self):
        for i in range(self._config.sim_updates_per_frame):
            self.Simulation.update(self._config.stdp_active, False)
            if self.Simulation.bupdate_chemical_contrations is True:
                self.chemical_concentrations.update_visuals()
