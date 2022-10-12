from dataclasses import dataclass
import numpy as np
import torch

from network import (
    NetworkInitValues,
    NetworkConfig,
    PlottingConfig,
    SpikingNeuralNetwork,
    NetworkGrid
)
from network.visualized_elements.boxes import (
    SelectedGroups,
    GroupInfo,
)
from network.visualized_elements import Neurons, VoltageMultiPlot, FiringScatterPlot, GroupFiringCountsPlot
from network.network_array_shapes import NetworkArrayShapes
from engine import EngineConfig, Engine
from network.gpu_arrays.network_gpu_arrays import NetworkGPUArrays


class IOSnn0GPU(NetworkGPUArrays):

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
                 voltage_multiplot: VoltageMultiPlot,
                 firing_scatter_plot: FiringScatterPlot,
                 selected_group_boxes,
                 group_firing_counts_plot_single1: GroupFiringCountsPlot
                 ):

        super().__init__(config=config, grid=grid, neurons=neurons, type_group_dct=type_group_dct,
                         type_group_conn_dct=type_group_conn_dct, device=device, T=T, shapes=shapes,
                         plotting_config=plotting_config, voltage_multiplot=voltage_multiplot,
                         firing_scatter_plot=firing_scatter_plot, selected_group_boxes=selected_group_boxes,
                         group_firing_counts_plot_single1=group_firing_counts_plot_single1)

        n0 = self.G_neuron_typed_ccount[67]

        self.N_flags.model[n0 + 2] = 1
        self.N_flags.model[n0 + 3] = 1
        self.N_flags.model[n0 + 4] = 1
        self.N_flags.model[n0 + 5] = 1

        self.synapse_arrays.N_rep[0, n0] = n0 + 2
        self.synapse_arrays.N_rep[0, n0 + 1] = n0 + 3
        self.synapse_arrays.N_rep[1, n0] = n0 + 4
        self.synapse_arrays.N_rep[1, n0 + 1] = n0 + 5

        self.synapse_arrays.make_sensory_groups(
            G_neuron_counts=self.G_neuron_counts, N_pos=self.N_pos, G_pos=self.G_pos,
            groups=self._config.sensory_groups, grid=grid)

        self.N_states.izhikevich_neurons.use_preset('RS', self.N_flags.select_by_groups(self._config.sensory_groups))

        if self._config.N >= 8000:
            self.synapse_arrays.swap_group_synapses(
                groups=torch.from_numpy(grid.forward_groups).to(device=self.device).type(torch.int64))

        self.Simulation.set_stdp_config(0)

        self._post_synapse_mod_init(shapes)

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

        self._group_firing_counts_plot_single1_tensor[
            t_mod: t_mod + n_updates, 1] = \
            self.G_firing_count_hist[t_mod: t_mod + n_updates, 123] / self.G_neuron_counts[1, 123]

        offset1 = self._plotting_config.scatter_plot_length

        self._group_firing_counts_plot_single1_tensor[
            offset1 + t_mod: offset1 + t_mod + n_updates, 1] = \
            self.G_firing_count_hist[t_mod: t_mod + n_updates, 125] / self.G_neuron_counts[1, 125]

        if t_mod + n_updates + 1 >= self._plotting_config.scatter_plot_length:
            self.G_firing_count_hist[:] = 0


class IOSnn0(SpikingNeuralNetwork):

    def __init__(self, config, app):

        super().__init__(config, app)

    # noinspection PyPep8Naming
    def initialize_GPU_arrays(self, device, engine: Engine):
        super().initialize_GPU_arrays(device, engine)

        self.voltage_plot = VoltageMultiPlot(
            scene=engine.voltage_multiplot_scene,
            view=engine.voltage_multiplot_view,
            device=device,
            n_plots=self.plotting_config.n_voltage_plots,
            plot_length=self.plotting_config.voltage_plot_length,
            n_group_separator_lines=self.data_shapes.n_group_separator_lines
        )

        self.firing_scatter_plot = FiringScatterPlot(
            scene=engine.firing_scatter_plot_scene,
            view=engine.firing_scatter_plot_view,
            device=device,
            n_plots=self.plotting_config.n_scatter_plots,
            plot_length=self.plotting_config.scatter_plot_length,
            n_group_separator_lines=self.data_shapes.n_group_separator_lines
        )

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

        self.selected_group_boxes = SelectedGroups(
            scene=engine.main_window.scene_3d,
            view=engine.main_window.scene_3d.network_view,
            network_config=self.network_config, grid=self.grid,
            connect=np.zeros((self.network_config.G + 1, 2)) + self.network_config.G,
            device=device,
        )

        self.GPU = IOSnn0GPU(
            config=self.network_config,
            grid=self.grid,
            neurons=self._neurons,
            type_group_dct=self.type_group_dct,
            type_group_conn_dct=self.type_group_conn_dict,
            device=device,
            T=self.T,
            shapes=self.data_shapes,
            plotting_config=self.plotting_config,
            voltage_multiplot=self.voltage_plot,
            firing_scatter_plot=self.firing_scatter_plot,
            selected_group_boxes=self.selected_group_boxes,
            group_firing_counts_plot_single1=self.group_firing_counts_plot_single1
        )

        self.registered_buffers += self.GPU.registered_buffers

        self.selected_group_boxes.init_cuda_attributes(G_flags=self.GPU.G_flags, G_props=self.GPU.G_props)

        self.group_info_mesh = GroupInfo(
            scene=engine.group_info_scene,
            view=engine.group_info_view,
            network_config=self.network_config, grid=self.grid,
            connect=np.zeros((self.network_config.G + 1, 2)) + self.network_config.G,
            device=device, G_flags=self.GPU.G_flags, G_props=self.GPU.G_props,
            g2g_info_arrays=self.GPU.g2g_info_arrays
        )
        engine.set_main_context_as_current()

        # self.CPU = NetworkCPUArrays(self.network_config, self.GPU)

        self.add_input_groups(scene=engine.main_window.scene_3d, view=engine.main_window.scene_3d.network_view)
        self.add_output_groups(scene=engine.main_window.scene_3d, view=engine.main_window.scene_3d.network_view)

        print('\nactive_sensory_groups:', self.GPU.active_sensory_groups)
        print('active_output_groups:', self.GPU.active_output_groups, '\n')
        
    def interface_single_neurons(self, engine: Engine):
        engine.interfaced_neuron_collection.add_interfaced_neuron(
            network=self, app=engine, window=engine.main_window,
            neuron_id=self.GPU.G_neuron_typed_ccount[67].item(), preset='tonic_bursting')
        engine.interfaced_neuron_collection.add_interfaced_neuron(
            network=self, app=engine, window=engine.main_window,
            neuron_id=self.GPU.G_neuron_typed_ccount[67].item()+1, preset='tonic_bursting')
        engine.interfaced_neuron_collection.add_interfaced_neuron(
            network=self, app=engine, window=engine.main_window,
            neuron_id=self.GPU.G_neuron_typed_ccount[67].item()+2, preset='low_pass_filter2')
        engine.interfaced_neuron_collection.add_interfaced_neuron(
            network=self, app=engine, window=engine.main_window,
            neuron_id=self.GPU.G_neuron_typed_ccount[67].item()+3, preset='low_pass_filter2')
        engine.interfaced_neuron_collection.add_interfaced_neuron(
            network=self, app=engine, window=engine.main_window,
            neuron_id=self.GPU.G_neuron_typed_ccount[67].item()+4, preset='high_pass_filter2')
        engine.interfaced_neuron_collection.add_interfaced_neuron(
            network=self, app=engine, window=engine.main_window,
            neuron_id=self.GPU.G_neuron_typed_ccount[67].item()+5, preset='high_pass_filter2')

        # engine.interfaced_neuron_collection.sync_signal(0, 1)
        # engine.interfaced_neuron_collection.sync_signal(2, 3)
        # engine.interfaced_neuron_collection.sync_model_variables(0, 2)
        engine.interfaced_neuron_collection.sync_model_variables(2, 3)
        engine.interfaced_neuron_collection.sync_model_variables(4, 5)

        engine.interfaced_neuron_collection.interfaced_neurons_dct['Neuron1'].current_control_frame.sliders.amplitude.set_prop_container_value(50)
        engine.interfaced_neuron_collection.interfaced_neurons_dct['Neuron1'].current_control_frame.sliders.amplitude.actualize_values()
        engine.interfaced_neuron_collection.interfaced_neurons_dct['Neuron2'].current_control_frame.sliders.amplitude.set_prop_container_value(0)
        engine.interfaced_neuron_collection.interfaced_neurons_dct['Neuron2'].current_control_frame.sliders.amplitude.actualize_values()
        engine.interfaced_neuron_collection.interfaced_neurons_dct['Neuron3'].current_control_frame.sliders.amplitude.set_prop_container_value(0)
        engine.interfaced_neuron_collection.interfaced_neurons_dct['Neuron3'].current_control_frame.sliders.amplitude.actualize_values()
        engine.interfaced_neuron_collection.interfaced_neurons_dct['Neuron4'].current_control_frame.sliders.amplitude.set_prop_container_value(0)
        engine.interfaced_neuron_collection.interfaced_neurons_dct['Neuron4'].current_control_frame.sliders.amplitude.actualize_values()
        engine.interfaced_neuron_collection.interfaced_neurons_dct['Neuron5'].current_control_frame.sliders.amplitude.set_prop_container_value(0)
        engine.interfaced_neuron_collection.interfaced_neurons_dct['Neuron5'].current_control_frame.sliders.amplitude.actualize_values()

    def unregister_registered_buffers(self):
        super().unregister_registered_buffers()

        self.voltage_plot.unregister_registered_buffers()
        self.firing_scatter_plot.unregister_registered_buffers()
        self.selected_group_boxes.unregister_registered_buffers()

        self.group_info_mesh.unregister_registered_buffers()

        self.group_firing_counts_multiplot.unregister_registered_buffers()
        self.group_firing_counts_plot_single0.unregister_registered_buffers()
        self.group_firing_counts_plot_single1.unregister_registered_buffers()


class IOSnn0Config(EngineConfig):

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

    N: int = 5 * 10 ** 3
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
