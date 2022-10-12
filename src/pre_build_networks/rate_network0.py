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


class RateNetwork0GPU(NetworkGPUArrays):

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

        self._post_synapse_mod_init(shapes)


class RateNetwork0(SpikingNeuralNetwork):

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

        engine.set_main_context_as_current()

        self.selected_group_boxes = SelectedGroups(
            scene=engine.main_window.scene_3d,
            view=engine.main_window.scene_3d.network_view,
            network_config=self.network_config, grid=self.grid,
            connect=np.zeros((self.network_config.G + 1, 2)) + self.network_config.G,
            device=device,
        )

        self.GPU = RateNetwork0GPU(
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

    def unregister_registered_buffers(self):
        super().unregister_registered_buffers()

        self.voltage_plot.unregister_registered_buffers()
        self.firing_scatter_plot.unregister_registered_buffers()
        self.selected_group_boxes.unregister_registered_buffers()

        self.group_info_mesh.unregister_registered_buffers()

class RateNetwork0Config(EngineConfig):

    N: int = 5 * 10 ** 3
    T: int = 5000  # Max simulation record duration

    device: int = 0

    max_batch_size_mb: int = 300

    network = NetworkConfig(N=N,
                            N_pos_shape=(4, 4, 1),
                            sim_updates_per_frame=1,
                            stdp_active=True,
                            debug=False, InitValues=EngineConfig.InitValues())

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

    network_class = RateNetwork0
