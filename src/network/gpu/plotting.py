# import torch
#
# from cpp_cuda_backend import GPUArrayCollection, RegisteredVBO
# from network.network_array_shapes import NetworkArrayShapes
# from network.network_config import BufferCollection, PlottingConfig
#
#
# class PlottingGPUArrays(GPUArrayCollection):
#
#     def __init__(self, plotting_config: PlottingConfig,
#                  device, shapes: NetworkArrayShapes,
#                  buffers: BufferCollection,
#                  bprint_allocated_memory,
#                  engine):
#
#         super().__init__(device=device, bprint_allocated_memory=bprint_allocated_memory)
#
#         self.buffers = buffers
#         self.registered_buffers = []
#         self.shapes = shapes
#
#         engine.main_window.scene_3d.set_current()
#
#         if buffers.group_firing_counts_plot_single0 is not None:
#             self.group_firing_counts_plot_single0 = RegisteredVBO(buffers.group_firing_counts_plot_single0,
#                                                                   (plotting_config.scatter_plot_length * 2, 2),
#                                                                   self.device)
#             self.registered_buffers.append(self.group_firing_counts_plot_single0)
#
#         if buffers.group_firing_counts_plot_single1 is not None:
#             self.group_firing_counts_plot_single1 = RegisteredVBO(buffers.group_firing_counts_plot_single1,
#                                                                   (plotting_config.scatter_plot_length * 2, 2),
#                                                                   self.device)
#             self.registered_buffers.append(self.group_firing_counts_plot_single1)
