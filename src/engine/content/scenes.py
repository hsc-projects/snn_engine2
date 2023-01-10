from copy import copy
from typing import Optional
import numpy as np

from vispy.app import Application
from vispy import scene
from vispy.visuals.transforms import STTransform
from vispy.util import keys

from .plot_widgets import (
    GroupInfoColorBar,
    GroupFiringsPlotWidget,
    VoltagePlotWidget,
    ScatterPlotWidget,
    SingleNeuronPlotWidget
)
from .widgets.base_plot_widget import BasePlotWidget

from network import SpikingNeuralNetwork
from rendering import RenderedObject
from network import PlottingConfig
from .widgets.scene_canvas_frame import CanvasConfig, TextTableWidget, BaseEngineSceneCanvas


class MainSceneCanvas(BaseEngineSceneCanvas):

    # noinspection PyTypeChecker
    def __init__(self,
                 conf: CanvasConfig,
                 app,
                 plotting_config: PlottingConfig):

        super().__init__(conf, app)

        self.unfreeze()

        self.network: Optional[SpikingNeuralNetwork] = None

        self.n_voltage_plots = plotting_config.n_voltage_plots
        self.voltage_plot_length = plotting_config.voltage_plot_length
        self.n_scatter_plots = plotting_config.n_scatter_plots
        self.scatter_plot_length = plotting_config.scatter_plot_length

        self.network_view = self.central_widget.add_view()

        self.grid: scene.widgets.Grid = self.network_view.add_grid()

        self.network_view.camera = 'turntable'
        axis = scene.visuals.XYZAxis(parent=self.network_view.scene)
        axis.transform = STTransform()
        axis.transform.move((-0.1, -0.1, -0.1))

        row_span_0 = 2
        col_span0 = 2
        plot_row0 = 0
        plot_col0 = 0
        plot_col1 = 4
        plot_col2 = plot_col1 + col_span0
        plot_row1 = plot_row0 + row_span_0
        row_span10 = 2
        row_span_11 = 1
        height_min0 = 350
        height_max1 = 500

        self.table = TextTableWidget(labels=['t', 'update_duration'])
        # self.info_grid_right(row=0, col=plot_col2 + 1, row_span=row_span_0, height_min=height_min0)
        self.grid.add_widget(self.table, 0, plot_col2+1)
        if plotting_config.windowed_multi_neuron_plots is False:
            if plotting_config.has_voltage_multiplot is True:
                self.voltage_plot = VoltagePlotWidget(plotting_confing=plotting_config,
                                                      width_max=600, height_min=height_min0)
                self.grid.add_widget(self.voltage_plot, plot_row0, plot_col0, row_span=row_span_0, col_span=col_span0)
            if plotting_config.has_firing_scatterplot is True:
                self.scatter_plot = ScatterPlotWidget(plotting_confing=plotting_config,
                                                      width_max=600, height_max=height_max1)
                self.grid.add_widget(self.scatter_plot, plot_row1, plot_col0, row_span=row_span10, col_span=col_span0)

        if plotting_config.group_info_view_mode.scene is True:
            if plotting_config.has_group_firings_multiplot is True:
                self.group_firings_multiplot = GroupFiringsPlotWidget(plotting_confing=plotting_config)
                self.grid.add_widget(self.group_firings_multiplot, plot_row1, plot_col1,
                                     col_span=col_span0, row_span=row_span10)
            self.color_bar = GroupInfoColorBar()
            self.grid.add_widget(self.color_bar, plot_row1, plot_col0 + col_span0, row_span10, 1)
        if plotting_config.has_group_firings_plot0:
            self.group_firings_plot_single0 = BasePlotWidget(
                title="Group Firings 0:",
                plot_height=1, plot_length=self.scatter_plot_length,
                cam_yscale=1)
            self.grid.add_widget(self.group_firings_plot_single0, plot_row1, plot_col2,
                                 col_span=col_span0, row_span=row_span_11)
        if plotting_config.has_group_firings_plot1:
            self.group_firings_plot_single1 = BasePlotWidget(
                title="Group Firings 1:",
                plot_height=1, plot_length=self.scatter_plot_length)
            self.grid.add_widget(self.group_firings_plot_single1, plot_row1 + row_span_11, plot_col2,
                                 col_span=2, row_span=row_span_11)

        self._clicked_obj = None
        self._selected_objects = []
        self._last_selected_obj = None

        self._click_pos = np.zeros(2)
        self._last_mouse_pos = np.zeros(2)

        self.mouse_pressed = True

        self.grid_transform = self.scene.node_transform(self.grid)

        self.freeze()

    @property
    def _window_id(self):
        # noinspection PyProtectedMember
        return self._backend._id

    def set_keys(self, keys_):
        self.unfreeze()
        # noinspection PyProtectedMember
        self._set_keys(keys_)
        self.freeze()

    def mouse_pos(self, event):
        return self.grid_transform.map(event.pos)[:2]

    def _select_clicked_obj(self):

        if self._clicked_obj is not self._last_selected_obj:

            self.network.neurons._neuron_visual.face_color[:, 3] = 0.05
            self.network.neurons._neuron_visual.edge_color[:, 3] = 0.05
            self.network.neurons._neuron_visual.set_gl_state(depth_test=False)

            for o in copy(self._selected_objects):
                if ((not (o is self._clicked_obj))
                        and ((self._clicked_obj is None) or (not o.is_select_child(self._clicked_obj)))):
                    self._select(o, False)

            if isinstance(self._clicked_obj, RenderedObject):
                if self._clicked_obj.selected:
                    # self.clicked_obj.on_select_callback(self.clicked_obj.selected)
                    self._clicked_obj.update()
                    self._last_selected_obj = self._clicked_obj
                else:
                    # print('\nSELECTED:', self._clicked_obj)
                    self._select(self._clicked_obj, True)

    def on_mouse_press(self, event):

        if event.button == 1:
            self.network_view.camera.interactive = False
            self.network_view.interactive = False
            self._clicked_obj = self.visual_at(event.pos)
            # print('\nCLICKED:', self._clicked_obj)
            self.network_view.interactive = True
            self._click_pos[:2] = self.mouse_pos(event)

            if isinstance(self._clicked_obj, RenderedObject) and self._clicked_obj.draggable:
                self._select_clicked_obj()

    def _mouse_moved(self, event):
        self._last_mouse_pos[:2] = self.mouse_pos(event)
        return (self._last_mouse_pos[:2] - self._click_pos[:2]).any()

    def _select(self, obj: RenderedObject, v: bool):
        obj.select(v)
        if v is True:
            self._selected_objects.append(obj)
            self._last_selected_obj = obj
        else:
            self._selected_objects.remove(obj)
            if obj is self._last_selected_obj:
                self._last_selected_obj = None
        return obj

    def on_mouse_release(self, event):
        self.network_view.camera.interactive = True
        if event.button == 1:
            if (not self._mouse_moved(event)) or (self._clicked_obj is self.visual_at(event.pos)):
                self._select_clicked_obj()
            if isinstance(self._last_selected_obj, RenderedObject) and self._last_selected_obj.draggable:
                self._select(self._last_selected_obj, False).update()
                self._last_selected_obj = self._selected_objects[-1]
            # self._last_selected_obj = None

            print(f'currently selected ({len(self._selected_objects)}):', self._selected_objects)
            if len(self._selected_objects) == 0:
                self.network.neurons._neuron_visual.face_color[:, 3] = 0.3
                self.network.neurons._neuron_visual.edge_color[:, 3] = 0.5
                self.network.neurons._neuron_visual.set_gl_state(depth_test=True)

    def on_mouse_move(self, event):
        self.network_view.camera.interactive = True
        if event.button == 1:
            if isinstance(self._clicked_obj, RenderedObject) and self._clicked_obj.draggable:
                # print(keys.SHIFT in event.modifiers)
                self.network_view.camera.interactive = False
                self._last_mouse_pos[:2] = self.mouse_pos(event)
                # dist = np.linalg.norm(self._last_mouse_pos - self._click_pos)
                diff = self._last_mouse_pos - self._click_pos
                # print('diff:', diff)
                if keys.SHIFT in event.modifiers:
                    mode = 0
                elif keys.CONTROL in event.modifiers:
                    mode = 1
                else:
                    mode = 2
                self._clicked_obj.on_drag_callback(diff/100, mode=mode)
                # self._clicked_obj.


class VoltagePlotSceneCanvas(BaseEngineSceneCanvas):

    # noinspection PyTypeChecker
    def __init__(self,
                 conf: CanvasConfig,
                 app: Optional[Application],
                 plotting_config: PlottingConfig,
                 display_time: bool = False):

        super().__init__(conf, app)

        self.unfreeze()
        self.n_voltage_plots = plotting_config.n_voltage_plots
        self.voltage_plot_length = plotting_config.voltage_plot_length

        main_grid: scene.widgets.Grid = self.central_widget.add_grid()
        self.central_widget.margin = 10

        self.plot = VoltagePlotWidget(plotting_confing=plotting_config, height_min=200)
        main_grid.add_widget(self.plot, row=0, row_span=9, col_span=4)
        if display_time is True:
            self.table = TextTableWidget(labels=['t'], height_max_global=25)
            self.table.height_max = 25
            main_grid.add_widget(self.table, 0, 3)
        self.freeze()


class ScatterPlotSceneCanvas(BaseEngineSceneCanvas):

    # noinspection PyTypeChecker
    def __init__(self,
                 conf: CanvasConfig,
                 app: Optional[Application],
                 plotting_config: PlottingConfig,
                 display_time: bool = False):

        super().__init__(conf, app)

        self.unfreeze()
        self.n_scatter_plots = plotting_config.n_scatter_plots
        self.scatter_plot_length = plotting_config.scatter_plot_length

        grid: scene.widgets.Grid = self.central_widget.add_grid()
        self.central_widget.margin = 10

        self.plot = ScatterPlotWidget(plotting_confing=plotting_config, height_min=200)

        grid.add_widget(self.plot, row=0, row_span=9, col_span=4)
        if display_time is True:
            self.table = TextTableWidget(labels=['t'], height_max_global=25)
            self.table.height_max = 25
            grid.add_widget(self.table, 0, 3)
        self.freeze()


class LocationGroupInfoCanvas(BaseEngineSceneCanvas):

    def __init__(self, conf: CanvasConfig, app: Optional[Application], plotting_config: PlottingConfig,
                 display_time: bool = False):

        super().__init__(conf, app)
        self.unfreeze()
        self.view = self.central_widget.add_view()
        self.view.camera = 'turntable'
        axis = scene.visuals.XYZAxis(parent=self.view.scene)
        axis.transform = STTransform()
        axis.transform.move((-0.1, -0.1, -0.1))

        self.grid = self.view.add_grid()

        if display_time is True:
            self.table = TextTableWidget(labels=['t'], height_max_global=25)
            self.table.height_max = 25
            self.grid.add_widget(self.table, 0, 6)

        if plotting_config.has_group_firings_multiplot:
            self.group_firings_multiplot = GroupFiringsPlotWidget(plotting_confing=plotting_config, width_max=600)
            self.grid.add_widget(self.group_firings_multiplot, 5, 5, col_span=2, row_span=6)

        self.color_bar = GroupInfoColorBar()
        self.grid.add_widget(self.color_bar, 5, 0, 6, 1)

        self.freeze()


class SingleNeuronPlotCanvas(BaseEngineSceneCanvas):
    def __init__(self,
                 conf: CanvasConfig,
                 app: Optional[Application],
                 plotting_config: PlottingConfig,
                 width_min=300, width_max=300, height_min=190, height_max=190):

        super().__init__(conf, app)

        self.unfreeze()

        self.plot_widget = SingleNeuronPlotWidget(plotting_confing=plotting_config,
                                                  width_min=width_min,
                                                  width_max=width_max,
                                                  height_min=height_min, height_max=height_max)
        grid = self.central_widget.add_grid()
        grid.add_widget(self.plot_widget)
        self.freeze()
