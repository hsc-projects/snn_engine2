import torch

from gpu import GPUArrayCollection, RegisteredVBO
from network.network_array_shapes import NetworkArrayShapes
from network.network_config import BufferCollection, PlottingConfig


class PlottingGPUArrays(GPUArrayCollection):

    def __init__(self, plotting_config: PlottingConfig,
                 device, shapes: NetworkArrayShapes,
                 buffers: BufferCollection,
                 bprint_allocated_memory,
                 app):

        super().__init__(device=device, bprint_allocated_memory=bprint_allocated_memory)

        self.buffers = buffers
        self.registered_buffers = []
        self.shapes = shapes

        self.voltage = None
        self.voltage_group_line_pos = None
        self.voltage_group_line_colors = None

        self.firings = None
        self.firings_group_line_pos = None
        self.firings_group_line_colors = None

        if app.neuron_plot_window is not None:
            app.neuron_plot_window.voltage_plot_sc.set_current()
        if self.buffers.voltage is not None:
            self.init_voltage_plot_arrays()

        if app.neuron_plot_window is not None:
            app.neuron_plot_window.scatter_plot_sc.set_current()
        if self.buffers.firings is not None:
            self.init_firings_plot_arrays()

        app.main_window.scene_3d.set_current()

        self.voltage_map = self.izeros(shapes.voltage_plot_map)
        self.voltage_map[:] = torch.arange(shapes.voltage_plot_map)

        self.firings_map = self.izeros(shapes.firings_scatter_plot_map)
        self.firings_map[:] = torch.arange(shapes.firings_scatter_plot_map)

        self.voltage_plot_slots = torch.arange(shapes.voltage_plot_map + 1, device=self.device)
        self.firings_plot_slots = torch.arange(shapes.firings_scatter_plot_map + 1, device=self.device)

        # print(self.voltage.to_dataframe)
        if buffers.group_firing_counts_plot_single0 is not None:
            self.group_firing_counts_plot_single0 = RegisteredVBO(buffers.group_firing_counts_plot_single0,
                                                                  (plotting_config.scatter_plot_length * 2, 2),
                                                                  self.device)
            self.registered_buffers.append(self.group_firing_counts_plot_single0)

        if buffers.group_firing_counts_plot_single1 is not None:
            self.group_firing_counts_plot_single1 = RegisteredVBO(buffers.group_firing_counts_plot_single1,
                                                                  (plotting_config.scatter_plot_length * 2, 2),
                                                                  self.device)
            self.registered_buffers.append(self.group_firing_counts_plot_single1)

    # noinspection DuplicatedCode
    def init_voltage_plot_arrays(self):
        self.voltage = RegisteredVBO(self.buffers.voltage, self.shapes.voltage_plot, self.device)

        self.voltage_group_line_pos = RegisteredVBO(self.buffers.voltage_group_line_pos,
                                                    self.shapes.plot_group_line_pos,
                                                    self.device)
        self.voltage_group_line_colors = RegisteredVBO(self.buffers.voltage_group_line_colors,
                                                       self.shapes.plot_group_line_colors,
                                                       self.device)
        self.registered_buffers.append(self.voltage)
        self.registered_buffers.append(self.voltage_group_line_pos)
        self.registered_buffers.append(self.voltage_group_line_colors)

    # noinspection DuplicatedCode
    def init_firings_plot_arrays(self):
        self.firings = RegisteredVBO(self.buffers.firings, self.shapes.firings_scatter_plot, self.device)

        self.firings_group_line_pos = RegisteredVBO(self.buffers.firings_group_line_pos,
                                                    self.shapes.plot_group_line_pos, self.device)

        self.firings_group_line_colors = RegisteredVBO(self.buffers.firings_group_line_colors,
                                                       self.shapes.plot_group_line_colors,
                                                       self.device)
        self.registered_buffers.append(self.firings)
        self.registered_buffers.append(self.firings_group_line_pos)
        self.registered_buffers.append(self.firings_group_line_colors)
