from dataclasses import dataclass
import torch

from network import (
    NetworkInitValues,
    NetworkConfig,
    PlottingConfig,
    SpikingNeuralNetwork,
)
from network.gpu.visualized_elements import GroupFiringCountsPlot
from engine import EngineConfig, Engine
from network.gpu.simulation import NetworkSimulationGPU
# from network.gpu.neurons import NeuronRepresentation
# from network.gpu.synapses import SynapseRepresentation


class IOSnn1GPU(NetworkSimulationGPU):

    def update(self):

        if self.Simulation.t % 100 == 0:
            print('t =', self.Simulation.t)
        if self.Simulation.t % 1000 == 0:
            # if False:
            self.Simulation.calculate_avg_group_weight()
            a = self.to_dataframe(self.neurons.g2g_info_arrays.G_avg_weight_inh)
            b = self.to_dataframe(self.neurons.g2g_info_arrays.G_avg_weight_exc)
            r = 6
            # self.look_up([(80, 72), (88, 80), (96, 88), (104, 96), (112, 104)],
            # self.G_stdp_config0.type(torch.float32))
            # self.look_up([(80, 72), (88, 80), (96, 88), (104, 96), (112, 104)], self.G_avg_weight_inh)
            # self.look_up([(80, 72), (88, 80), (96, 88), (104, 96), (112, 104)], self.G_avg_weight_exc)
            # print()
            self.look_up([(75, 67), (83, 75), (91, 83), (99, 91), (107, 99), (115, 107), (123, 115)],
                         self.neurons.g2g_info_arrays.G_avg_weight_inh)
            self.look_up([(75, 67), (83, 75), (91, 83), (99, 91), (107, 99), (115, 107), (123, 115)],
                         self.neurons.g2g_info_arrays.G_avg_weight_exc)
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

        self._group_firing_counts_plot_single1_tensor[
            t_mod: t_mod + n_updates, 1] = \
            self.G_firing_count_hist[t_mod: t_mod + n_updates, 123] / \
            self.neurons.G_neuron_counts[1, 123]

        offset1 = self._plotting_config.scatter_plot_length

        self._group_firing_counts_plot_single1_tensor[
            offset1 + t_mod: offset1 + t_mod + n_updates, 1] = \
            self.G_firing_count_hist[t_mod: t_mod + n_updates, 125] / \
            self.neurons.G_neuron_counts[1, 125]

        if t_mod + n_updates + 1 >= self._plotting_config.scatter_plot_length:
            self.G_firing_count_hist[:] = 0


class IOSnn0(SpikingNeuralNetwork):

    def __init__(self, config, engine):

        super().__init__(config, engine)

    # noinspection PyPep8Naming
    def initialize_GPU_arrays(self, device, engine: Engine, **kwargs):
        super().initialize_GPU_arrays(device, engine)

        self.group_firing_counts_multiplot = GroupFiringCountsPlot(
            scene=engine.group_info_scene,
            view=engine.group_firings_multiplot_view,
            device=device,
            n_plots=self.network_config.G,
            plot_length=self.plotting_config.scatter_plot_length,
            n_groups=self.network_config.G)

        self.group_firing_counts_plot_single0 = GroupFiringCountsPlot(
            scene=engine.main_window.scene_3d,
            view=engine.main_window.scene_3d.group_firings_plot_single0,
            device=device,
            n_plots=2,
            plot_length=self.plotting_config.scatter_plot_length,
            n_groups=0)

        self.group_firing_counts_plot_single1 = GroupFiringCountsPlot(
            scene=engine.main_window.scene_3d,
            view=engine.main_window.scene_3d.group_firings_plot_single1,
            device=device,
            n_plots=2,
            plot_length=self.plotting_config.scatter_plot_length,
            n_groups=0,
            color=[[1., 0., 0., 1.], [0., 1., 0., 1.]]
        )

        engine.set_main_context_as_current()

        self.simulation_gpu = IOSnn1GPU.from_snn(self, engine=engine, device=device)

        n0 = self.neurons.G_neuron_typed_ccount[67]

        self.neurons.N_flags.model[n0 + 2] = 1
        self.neurons.N_flags.model[n0 + 3] = 1
        self.neurons.N_flags.model[n0 + 4] = 1
        self.neurons.N_flags.model[n0 + 5] = 1

        self.synapse_arrays.N_rep[0, n0] = n0 + 2
        self.synapse_arrays.N_rep[0, n0 + 1] = n0 + 3
        self.synapse_arrays.N_rep[1, n0] = n0 + 4
        self.synapse_arrays.N_rep[1, n0 + 1] = n0 + 5

        self.synapse_arrays.make_sensory_groups(
            G_neuron_counts=self.neurons.G_neuron_counts,
            N_pos=self.neurons._neuron_visual.gpu_array,
            G_pos=self.neurons.G_pos,
            groups=self.network_config.sensory_groups,
            grid=self.network_config.grid)

        self.neurons.N_states.izhikevich_neurons.use_preset(
            'RS', self.neurons.N_flags.select_by_groups(self.network_config.sensory_groups))

        if self.network_config.N >= 8000:
            self.synapse_arrays.swap_group_synapses(
                groups=torch.from_numpy(self.simulation_gpu._config.grid.forward_groups).to(
                    device=self.simulation_gpu.device).type(torch.int64))

        self.simulation_gpu.Simulation.set_stdp_config(0)

        self.simulation_gpu._post_synapse_mod_init()

        self.registered_buffers += self.simulation_gpu.registered_buffers

        # self.CPU = NetworkCPUArrays(self.network_config, self.GPU)

        self.add_input_groups(scene=engine.main_window.scene_3d, view=engine.main_window.scene_3d.network_view)
        self.add_output_groups(scene=engine.main_window.scene_3d, view=engine.main_window.scene_3d.network_view)

        print('\nactive_sensory_groups:', self.neurons.active_sensory_groups)
        print('active_output_groups:', self.neurons.active_output_groups, '\n')
        
    def interface_single_neurons(self, engine: Engine):
        engine.interfaced_neuron_collection.add_interfaced_neuron(
            network=self, app=engine, window=engine.main_window,
            neuron_id=self.neurons.G_neuron_typed_ccount[67].item(), preset='tonic_bursting')
        engine.interfaced_neuron_collection.add_interfaced_neuron(
            network=self, app=engine, window=engine.main_window,
            neuron_id=self.neurons.G_neuron_typed_ccount[67].item() + 1, preset='tonic_bursting')
        engine.interfaced_neuron_collection.add_interfaced_neuron(
            network=self, app=engine, window=engine.main_window,
            neuron_id=self.neurons.G_neuron_typed_ccount[67].item() + 2, preset='low_pass_filter2')
        engine.interfaced_neuron_collection.add_interfaced_neuron(
            network=self, app=engine, window=engine.main_window,
            neuron_id=self.neurons.G_neuron_typed_ccount[67].item() + 3, preset='low_pass_filter2')
        engine.interfaced_neuron_collection.add_interfaced_neuron(
            network=self, app=engine, window=engine.main_window,
            neuron_id=self.neurons.G_neuron_typed_ccount[67].item() + 4, preset='high_pass_filter2')
        engine.interfaced_neuron_collection.add_interfaced_neuron(
            network=self, app=engine, window=engine.main_window,
            neuron_id=self.neurons.G_neuron_typed_ccount[67].item() + 5, preset='high_pass_filter2')

        # engine.interfaced_neuron_collection.sync_signal(0, 1)
        # engine.interfaced_neuron_collection.sync_signal(2, 3)
        # engine.interfaced_neuron_collection.sync_model_variables(0, 2)
        engine.interfaced_neuron_collection.sync_model_variables(2, 3)
        engine.interfaced_neuron_collection.sync_model_variables(4, 5)

        interfaced_neurons_dct = engine.interfaced_neuron_collection.interfaced_neurons_dct

        interfaced_neurons_dct['Neuron1'].current_control_frame.sliders.amplitude.set_prop_container_value(50)
        interfaced_neurons_dct['Neuron1'].current_control_frame.sliders.amplitude.actualize_values()
        interfaced_neurons_dct['Neuron2'].current_control_frame.sliders.amplitude.set_prop_container_value(0)
        interfaced_neurons_dct['Neuron2'].current_control_frame.sliders.amplitude.actualize_values()
        interfaced_neurons_dct['Neuron3'].current_control_frame.sliders.amplitude.set_prop_container_value(0)
        interfaced_neurons_dct['Neuron3'].current_control_frame.sliders.amplitude.actualize_values()
        interfaced_neurons_dct['Neuron4'].current_control_frame.sliders.amplitude.set_prop_container_value(0)
        interfaced_neurons_dct['Neuron4'].current_control_frame.sliders.amplitude.actualize_values()
        interfaced_neurons_dct['Neuron5'].current_control_frame.sliders.amplitude.set_prop_container_value(0)
        interfaced_neurons_dct['Neuron5'].current_control_frame.sliders.amplitude.actualize_values()

    def unregister_registered_buffers(self):
        super().unregister_registered_buffers()

        self.group_firing_counts_multiplot.unregister_registered_buffers()
        self.group_firing_counts_plot_single0.unregister_registered_buffers()
        self.group_firing_counts_plot_single1.unregister_registered_buffers()


class IOSnn1Config(EngineConfig):

    class InitValues(NetworkInitValues):

        @dataclass
        class ThalamicInput:
            inh_current: float = 25.
            exc_current: float = 15.

        @dataclass
        class SensoryInput:
            input_current0: float = 22.3
            input_current1: float = 42.8

        @dataclass
        class Weights:
            Inh2Exc: float = -.49
            Exc2Inh: float = .75
            Exc2Exc: float = .75
            SensorySource: float = 1.5

    N: int = 2 * 10 ** 5
    T: int = 5000  # Max simulation record duration

    device: int = 0

    max_batch_size_mb: int = 300

    network = NetworkConfig(N=N,
                            N_pos_shape=(4, 4, 1),
                            sim_updates_per_frame=1,
                            stdp_active=True,
                            debug=False, InitValues=InitValues())

    plotting = PlottingConfig(n_voltage_plots=10,
                              voltage_plot_length=200,
                              n_scatter_plots=10,
                              scatter_plot_length=200,
                              has_voltage_multiplot=True,
                              has_firing_scatterplot=True,
                              has_group_firings_multiplot=True,
                              has_group_firings_plot0=True,
                              has_group_firings_plot1=True,
                              windowed_multi_neuron_plots=False,
                              windowed_neuron_interfaces=True,
                              group_info_view_mode='split',
                              network_config=network)

    network_class = IOSnn0
    update_single_neuron_plots: bool = True
