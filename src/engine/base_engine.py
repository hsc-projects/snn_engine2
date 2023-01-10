import qdarktheme
import numba.cuda
from PyQt6.QtWidgets import QApplication
# from typing import Optional
from vispy.app import Application, Timer

from .windows import (
    MainWindow,
    MultiNeuronPlotWindow,
    LocationGroupInfoWindow,
    MainUILeft,
    NeuronInterfaceWindow
)
# from network import SpikingNeuralNetwork
from network.network_config import PlottingConfig
from .base_engine_config import EngineConfig


class Engine(Application):

    def __init__(self, config):

        self.config: EngineConfig = config

        self.native_app = QApplication([''])

        screens = self.native_app.screens()

        super().__init__(backend_name='pyqt6')

        # self.network: Optional[SpikingNeuralNetwork] = None

        self._plotting_config: PlottingConfig = self.config.plotting
        self._group_info_view_mode = self._plotting_config.group_info_view_mode

        self.native_app.setStyleSheet(qdarktheme.load_stylesheet())

        self.main_window: MainWindow = MainWindow(name="SNN Engine", app=self, plotting_config=self._plotting_config)
        self.main_window.setScreen(screens[0])

        if self._group_info_view_mode.split is True:
            self.main_window.add_group_info_scene_to_splitter(self._plotting_config)

        self.main_window.show()
        self.main_window.setGeometry(screens[0].availableGeometry())

        if self._plotting_config.windowed_neuron_interfaces is True:
            self.interfaced_neurons_window = NeuronInterfaceWindow(parent=self.main_window)
            self.interfaced_neuron_collection = self.interfaced_neurons_window.panel
            self.interfaced_neurons_window.show()
        else:
            self.interfaced_neuron_collection = self.main_window.ui_panel_left.neurons_collapsible
        # keep order for vbo numbers (2/3)

        if self._plotting_config.windowed_multi_neuron_plots is True:
            # self.neuron_plot_window: MultiNeuronPlotWindow = self._init_neuron_plot_window()
            self.neuron_plot_window = MultiNeuronPlotWindow(app=self, plotting_config=self._plotting_config)
            self.neuron_plot_window.show()
        else:
            self.neuron_plot_window = None

        if self._group_info_view_mode.windowed is True:
            self.location_group_info_window = LocationGroupInfoWindow(
                app=self, parent=self.main_window, plotting_config=self._plotting_config)
            self.location_group_info_window.show()
            self.group_info_panel = self.location_group_info_window.ui_panel_left
        else:
            self.group_info_panel = self.main_window.ui_right

        self.time_elapsed_until_last_off = 0
        self.update_switch = False
        self.started = False
        self.timer_on = Timer('auto', connect=self.update, start=False)

        self.last_g_flags_text = None
        self.last_g_props_text = None
        self.last_g2g_info_text = None

        # keep order for vbo id (1/2)
        # noinspection PyUnresolvedReferences
        from pycuda import autoinit
        numba.cuda.select_device(self.config.device)
        self.network = self.config.network_class(self.config, self)
        # keep order (3/3)
        self._bind_ui()

    @property
    def actions(self):
        return self.main_window.menubar.actions

    @property
    def main_ui_panel(self) -> MainUILeft:
        return self.main_window.ui_panel_left

    @property
    def firing_scatter_plot_scene(self):
        if self._plotting_config.windowed_multi_neuron_plots is True:
            return self.neuron_plot_window.scatter_plot_sc
        else:
            return self.main_window.scene_3d

    @property
    def firing_scatter_plot_view(self):
        if self._plotting_config.windowed_multi_neuron_plots is True:
            return self.neuron_plot_window.scatter_plot_sc.plot.view
        else:
            return self.main_window.scene_3d.scatter_plot

    @property
    def group_firings_multiplot_view(self):
        if self._group_info_view_mode.windowed is True:
            return self.location_group_info_window.scene_3d.group_firings_multiplot
        elif self._group_info_view_mode.split is True:
            return self.main_window.group_info_scene.group_firings_multiplot
        elif self._group_info_view_mode.scene is True:
            return self.main_window.scene_3d.group_firings_multiplot

    @property
    def group_info_scene(self):
        if self._group_info_view_mode.windowed is True:
            return self.location_group_info_window.scene_3d
        elif self._group_info_view_mode.split is True:
            return self.main_window.group_info_scene
        elif self._group_info_view_mode.scene is True:
            return self.main_window.scene_3d

    @property
    def group_info_view(self):
        if self._group_info_view_mode.windowed is True:
            return self.location_group_info_window.scene_3d.view
        elif self._group_info_view_mode.split is True:
            return self.main_window.group_info_scene.view
        elif self._group_info_view_mode.scene is True:
            return self.main_window.scene_3d.network_view

    @property
    def voltage_multiplot_scene(self):
        if self._plotting_config.windowed_multi_neuron_plots is True:
            return self.neuron_plot_window.voltage_plot_sc
        else:
            return self.main_window.scene_3d

    @property
    def voltage_multiplot_view(self):
        if self._plotting_config.windowed_multi_neuron_plots is True:
            return self.neuron_plot_window.voltage_plot_sc.plot.view
        else:
            return self.main_window.scene_3d.voltage_plot

    def add_selector_box(self):
        s = self.network.add_selector_box(
            self.main_window.scene_3d, self.main_window.scene_3d.network_view)
        self.main_ui_panel.add_3d_object_sliders(s)

    def add_synapsevisual(self):
        self.main_ui_panel.synapse_collapsible.add_interfaced_synapse(self.network, 0)

    def set_main_context_as_current(self):
        self.main_window.scene_3d.set_current()

    def _bind_ui(self):

        self.main_window.scene_3d.network = self.network
        # noinspection PyUnresolvedReferences
        self.native_app.aboutToQuit.connect(self.network.unregister_registered_buffers)
        # noinspection PyUnresolvedReferences
        self.native_app.aboutToQuit.connect(self.interfaced_neuron_collection.unregister_registered_buffers)

        network_config = self.network.network_config

        self._connect_main_buttons_and_actions()

        if self.network.input_groups is not None:
            self.main_ui_panel.add_3d_object_sliders(self.network.input_groups)
            self.main_ui_panel.sliders.sensory_weight.connect_property(
                self.network.input_groups,
                self.network.input_groups.src_weight)
        if self.network.output_groups is not None:
            self.main_ui_panel.add_3d_object_sliders(self.network.output_groups)

        self._connect_g_props_sliders(network_config)

        self.connect_group_info_combo_box()

        self.network.interface_single_neurons(self)

        # self.main_ui_panel.synapse_collapsible.add_interfaced_synapse(self.network, 1)

        if self.network.chemical_concentrations is not None:
            self.main_ui_panel.chemical_collapsible.add_chemical_control_collapsibles(self.network.chemical_concentrations)

    def connect_chemical_controls(self):
        pass

    def connect_group_info_combo_box(self):

        group_info_mesh = self.network.simulation_gpu.group_info_mesh

        self.group_info_panel.group_ids_combobox().add_items(group_info_mesh.group_id_texts.keys())
        self.group_info_panel.g_flags_combobox().add_items(group_info_mesh.G_flags_texts.keys())
        self.group_info_panel.g_props_combobox().add_items(group_info_mesh.G_props_texts.keys())
        g_txt = group_info_mesh.group_id_texts[group_info_mesh.group_id_key]
        self.group_info_panel.g2g_info_combo_box.init_src_group_combo_box(g_txt)
        self.group_info_panel.g2g_info_combo_box().add_items(group_info_mesh.G2G_info_texts.keys())

        self.group_info_panel.group_ids_combobox.connect(self.group_id_combo_box_text_changed)
        self.group_id_combo_box_text_changed(group_info_mesh.group_id_key)
        self.group_info_panel.g_flags_combobox.connect(self.g_flags_combo_box_text_changed)
        self.group_info_panel.g_props_combobox.connect(self.g_props_combo_box_text_changed)
        self.group_info_panel.g2g_info_combo_box.connect(self.g2g_info_combo_box_text_changed)

        self.actions.actualize_g_flags.triggered.connect(self.g_flags_combo_box_text_changed)
        self.actions.actualize_g_props.triggered.connect(self.g_props_combo_box_text_changed)
        self.actions.actualize_g2g_info.triggered.connect(self.g2g_info_combo_box_text_changed)

        self.actions.toggle_groups_ids.triggered.connect(self.toggle_group_id_text)
        self.actions.toggle_g_flags.triggered.connect(self.toggle_g_flags_text)
        self.actions.toggle_g_props.triggered.connect(self.toggle_g_props_text)
        self.actions.toggle_g2g_info.triggered.connect(self.toggle_g2g_info_text)

        self.actions.toggle_groups_ids.setChecked(True)
        self.actions.toggle_g_flags.setChecked(True)
        self.actions.toggle_g_props.setChecked(True)
        self.actions.toggle_g2g_info.setChecked(True)

        self.last_g_flags_text = self.group_info_panel.g_flags_combobox().currentText()
        self.last_g_props_text = self.group_info_panel.g_props_combobox().currentText()
        self.last_g2g_info_text = self.group_info_panel.g2g_info_combo_box().currentText()

        self.group_info_panel.combo_boxes_collapsible0.toggle_collapsed()
        self.group_info_panel.combo_boxes_collapsible1.toggle_collapsed()

    def _connect_g_props_sliders(self, network_config):
        self.main_ui_panel.sliders.thalamic_inh_input_current.connect_property(
            self.network.neurons.G_props,
            network_config.InitValues.ThalamicInput.inh_current)
        self.main_ui_panel.sliders.thalamic_exc_input_current.connect_property(
            self.network.neurons.G_props,
            network_config.InitValues.ThalamicInput.exc_current)
        self.main_ui_panel.sliders.sensory_input_current0.connect_property(
            self.network.neurons.G_props,
            network_config.InitValues.SensoryInput.input_current0)
        self.main_ui_panel.sliders.sensory_input_current1.connect_property(
            self.network.neurons.G_props,
            network_config.InitValues.SensoryInput.input_current1)

    def _connect_main_buttons_and_actions(self):
        self.main_ui_panel.buttons.start.clicked.connect(self.trigger_update_switch)
        self.main_ui_panel.buttons.pause.clicked.connect(self.trigger_update_switch)
        self.main_ui_panel.buttons.exit.clicked.connect(self.quit)
        self.main_ui_panel.buttons.toggle_outergrid.clicked.connect(self.toggle_outergrid)

        self.main_ui_panel.buttons.add_selector_box.clicked.connect(self.add_selector_box)
        self.main_ui_panel.buttons.add_synapsevisual.clicked.connect(self.add_synapsevisual)

        self.actions.start.triggered.connect(self.trigger_update_switch)
        self.actions.pause.triggered.connect(self.trigger_update_switch)
        self.actions.exit.triggered.connect(self.quit)
        self.actions.toggle_outergrid.triggered.connect(self.toggle_outergrid)

        self.actions.add_selector_box.triggered.connect(self.add_selector_box)
        self.actions.add_synapsevisual.triggered.connect(self.add_synapsevisual)

    def group_id_combo_box_text_changed(self, s):
        print(s)
        clim = self.network.simulation_gpu.group_info_mesh.set_group_id_text(s)
        self._set_g2g_color_bar_clim(clim)

    @staticmethod
    def _toggle_group_info_text(combobox, last_value_attr):
        t = combobox.currentText()
        if t != 'None':
            combobox.setCurrentText('None')
        else:
            combobox.setCurrentText(last_value_attr)

    def toggle_group_id_text(self):
        self._toggle_group_info_text(self.group_info_panel.group_ids_combobox(),
                                     self.network.simulation_gpu.group_info_mesh.group_id_key)

    def toggle_g_flags_text(self):
        self._toggle_group_info_text(self.group_info_panel.g_flags_combobox(), self.last_g_flags_text)

    def toggle_g_props_text(self):
        self._toggle_group_info_text(self.group_info_panel.g_props_combobox(), self.last_g_props_text)

    def toggle_g2g_info_text(self):
        self._toggle_group_info_text(self.group_info_panel.g2g_info_combo_box(), self.last_g2g_info_text)

    def _set_g2g_color_bar_clim(self, clim):
        if clim[0] is not None:
            self.group_info_scene.color_bar.clim = clim

    def g_flags_combo_box_text_changed(self, s=None):
        if not s:
            s = self.group_info_panel.g_flags_combobox().currentText()
        if s != 'None':
            self.actions.toggle_g_flags.setChecked(True)
            self.last_g_flags_text = s
        else:

            self.actions.toggle_g_flags.setChecked(False)
        print(s)
        clim = self.network.simulation_gpu.group_info_mesh.set_g_flags_text(s)
        self._set_g2g_color_bar_clim(clim)

    def g_props_combo_box_text_changed(self, s):
        if not s:
            s = self.group_info_panel.g_props_combobox().currentText()
        if s != 'None':
            self.actions.toggle_g_props.setChecked(True)
            self.last_g_props_text = s
        else:
            self.actions.toggle_g_props.setChecked(False)
        print(s)
        clim = self.network.simulation_gpu.group_info_mesh.set_g_props_text(s)
        self._set_g2g_color_bar_clim(clim)

    def g2g_info_combo_box_text_changed(self, s):
        g = self.group_info_panel.g2g_info_combo_box.combo_box0.currentText()
        t = self.group_info_panel.g2g_info_combo_box().currentText()
        if t != 'None':
            self.actions.toggle_g_props.setChecked(True)
            self.last_g2g_info_text = t
        # else:
            self.actions.toggle_g_props.setChecked(False)
        print(g, t)
        clim = self.network.simulation_gpu.group_info_mesh.set_g2g_info_txt(int(g), t)
        self._set_g2g_color_bar_clim(clim)

    def toggle_outergrid(self):
        # self.add_selector_box()

        self.network.outer_grid.visible = not self.network.outer_grid.visible

        if self.network.outer_grid.visible is True:
            self.main_ui_panel.buttons.toggle_outergrid.setChecked(True)
            self.actions.toggle_outergrid.setChecked(True)

            if self._group_info_view_mode.scene is True:
                self.main_window.scene_3d.group_firings_multiplot.visible = True
            if self.neuron_plot_window is not None:
                self.neuron_plot_window.hide()

        else:
            self.main_ui_panel.buttons.toggle_outergrid.setChecked(False)
            self.actions.toggle_outergrid.setChecked(False)
            if self._group_info_view_mode.scene is True:
                self.main_window.scene_3d.group_firings_multiplot.visible = False
            if self.neuron_plot_window is not None:
                self.neuron_plot_window.show()

    def trigger_update_switch(self):
        self.update_switch = not self.update_switch
        if self.update_switch is True:
            self.timer_on.start()
            self.main_ui_panel.buttons.start.setDisabled(True)
            self.actions.start.setDisabled(True)
            self.main_ui_panel.buttons.pause.setDisabled(False)
            self.actions.pause.setDisabled(False)
        else:
            self.time_elapsed_until_last_off += self.timer_on.elapsed
            self.timer_on.stop()
            self.main_ui_panel.buttons.start.setDisabled(False)
            self.actions.start.setDisabled(False)
            self.main_ui_panel.buttons.pause.setDisabled(True)
            self.actions.pause.setDisabled(True)

    # noinspection PyUnusedLocal
    def update(self, event):
        if self.update_switch is True:
            self.network.simulation_gpu.update()
            t = self.network.simulation_gpu.Simulation.t
            t_str = str(t)
            if self.neuron_plot_window:
                self.neuron_plot_window.voltage_plot_sc.update()
                self.neuron_plot_window.scatter_plot_sc.update()
                # self.neuron_plot_window.voltage_plot_sc.table.t.text = t
                # self.neuron_plot_window.scatter_plot_sc.table.t.text = t
            if self._group_info_view_mode.split is True:
                self.main_window.group_info_scene.update()
                # self.main_window.group_info_scene.table.t.text = t
            self.main_window.scene_3d.table.t.text = t_str
            self.main_window.scene_3d.table.update_duration.text = str(
                self.network.simulation_gpu.Simulation.update_duration)

            if self.config.update_single_neuron_plots is True:
                self.interfaced_neuron_collection.update_interfaced_neuron_plots(t)
