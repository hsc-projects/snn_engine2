from dataclasses import dataclass

from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QLabel,
    QSpinBox
)
from typing import Optional, Union, Type

from engine.content.widgets.collapsible_widget.collapsible_widget import CollapsibleWidget
from .widgets.spin_box_sliders import SpinBoxSlider, SubCollapsibleFrame, SliderCollection
from engine.content.widgets.combobox_frame import ComboBoxFrame
from .scenes import SingleNeuronPlotCanvas
from interfaces import NeuronInterface

from network.network_state import (
    MultiModelNeuronStateTensor,
    IzhikevichModel,
    RulkovModel,
    StateRow)
from engine.content.widgets.scene_canvas_frame import SceneCanvasFrame, CanvasConfig
from signaling import SignalModel, BaseSignalVariable
from network import SpikingNeuralNetwork, PlottingConfig


class NeuronIDFrame(SubCollapsibleFrame):

    def __init__(self, parent, N: int, fixed_width=300, label='ID'):

        super().__init__(parent, fixed_width=fixed_width)

        self.layout().addWidget(QLabel(label))
        self.spinbox = QSpinBox(self)
        self.layout().addWidget(self.spinbox)
        self.spinbox.setMinimum(0)
        self.spinbox.setMaximum(N-1)
        self.setFixedHeight(28)


class NeuronPropertiesFrame(SubCollapsibleFrame):

    @dataclass
    class Sliders(SliderCollection):

        # TODO: dynamic
        pt: Optional[SpinBoxSlider] = None
        a: Optional[SpinBoxSlider] = None
        b: Optional[SpinBoxSlider] = None
        c: Optional[SpinBoxSlider] = None
        d: Optional[SpinBoxSlider] = None
        theta: Optional[SpinBoxSlider] = None
        kappa: Optional[SpinBoxSlider] = None
        epsilon: Optional[SpinBoxSlider] = None
        gamma: Optional[SpinBoxSlider] = None

    def __init__(self, parent, window,
                 interface: NeuronInterface,
                 fixed_width=300,
                 neuron_model: Union[Type[MultiModelNeuronStateTensor],
                                     Type[RulkovModel],
                                     Type[IzhikevichModel]] = MultiModelNeuronStateTensor
                 ):

        super().__init__(parent, fixed_width=fixed_width)
        self.neuron_interface = interface

        self.preset_combo_box_frame = ComboBoxFrame(
            'preset', max_width=300,
            item_list=list(self.neuron_interface.preset_dct.keys()))
        self.preset_combo_box_frame.connect(self.preset_changed)

        self.sliders = self.Sliders(parent=self)
        self.sliders.add(self.preset_combo_box_frame)

        for x in self.sliders.keys:
            row_def: StateRow = getattr(neuron_model.Rows, x)
            self.sliders.add_slider(
                x, self.neuron_interface,
                name=x + ':',
                window=window,
                _min_value=row_def.interval[0],
                _max_value=row_def.interval[1],
                boxlayout_orientation=QtCore.Qt.Orientation.Horizontal,
                prop_id=x,
                single_step_spin_box=row_def.step_size,
                single_step_slider=row_def.step_size * 1000)
            if x not in self.neuron_interface.model_variables:
                self.sliders.hide_slider(x)
        self.setFixedHeight(self.sliders.widget.maximumHeight() + 5)
        self.layout().addWidget(self.sliders.widget)

    def actualize_values(self, keys):
        # print(keys)
        for k in self.sliders.keys:
            slider: SpinBoxSlider = getattr(self.sliders, k)
            if k in keys:
                self.sliders.show_slider(k)
                slider.actualize_values()
            else:
                self.sliders.hide_slider(k)

        self.setFixedHeight(self.sliders.widget.maximumHeight() + 5)

    def preset_changed(self, v):
        if v != '':
            self.neuron_interface.use_preset(v)
            self.actualize_values(keys=self.neuron_interface.model_variables)

    def reset_preset_combobox(self):
        self.preset_combo_box_frame.combo_box.clear()
        self.preset_combo_box_frame.combo_box.add_items(list(self.neuron_interface.preset_dct.keys()))


class CurrentControlFrame(SubCollapsibleFrame):

    @dataclass
    class Sliders(SliderCollection):
        amplitude: Optional[SpinBoxSlider] = None
        step_time: Optional[SpinBoxSlider] = None
        period: Optional[SpinBoxSlider] = None
        duty: Optional[SpinBoxSlider] = None
        duty_period: Optional[SpinBoxSlider] = None
        offset: Optional[SpinBoxSlider] = None
        phase: Optional[SpinBoxSlider] = None
        frequency: Optional[SpinBoxSlider] = None
        spike_period: Optional[SpinBoxSlider] = None

    def __init__(self, parent, model: SignalModel, window, fixed_width=300):

        super().__init__(parent, fixed_width=fixed_width)

        self.sliders = self.Sliders(parent=self)

        for x in self.sliders.keys:
            if hasattr(model.VariableConfig, x):
                var_conf: BaseSignalVariable = getattr(model.VariableConfig, x)
                self.sliders.add_slider(
                    x, model,
                    name=x + ':',
                    window=window,
                    _min_value=var_conf.interval[0],
                    _max_value=var_conf.interval[1],
                    boxlayout_orientation=QtCore.Qt.Orientation.Horizontal,
                    prop_id=x,
                    single_step_spin_box=var_conf.step_size,
                    single_step_slider=var_conf.step_size * 1000,
                    suffix=f' [{var_conf.unit}]')

        self.setFixedHeight(self.sliders.widget.maximumHeight() + 5)
        self.layout().addWidget(self.sliders.widget)


class SingleNeuronPlotCollapsible(CollapsibleWidget):

    def __init__(self, parent, app, network, interface: NeuronInterface, title='plot'):

        super().__init__(parent, title=title)

        width_min = 200
        width_max = 850
        height_min = 130
        height_max = 130
        self.canvas = SingleNeuronPlotCanvas(
            conf=CanvasConfig(), app=app, plotting_config=network.plotting_config,
            width_min=width_min, width_max=width_max,
            height_min=height_min, height_max=height_max
        )

        interface.link_plot_widget(self.canvas.plot_widget)
        plot_frame = SceneCanvasFrame(self, self.canvas)
        plot_frame.setFixedSize(width_max+80, height_max+50)

        self.add(plot_frame)

        self.canvas.set_current()
        interface.register_vbos()


class SingleNeuronCollapsible(CollapsibleWidget):

    def __init__(self, parent, network: SpikingNeuralNetwork,
                 title, window, app,
                 neuron_model: Union[Type[IzhikevichModel],
                                     Type[RulkovModel],
                                     Type[MultiModelNeuronStateTensor]] = MultiModelNeuronStateTensor):

        super().__init__(parent=parent, title=title)

        self.neuron_interface = NeuronInterface(0, network)

        self.id = NeuronIDFrame(self, network.network_config.N)
        self.model_collapsible = CollapsibleWidget(self, title='model')
        self.model_collapsible._title_frame.layout()

        self.model_frame = NeuronPropertiesFrame(self.model_collapsible, window,
                                                 interface=self.neuron_interface, neuron_model=neuron_model)
        self.add(self.id)
        self.model_collapsible.add(self.model_frame)
        self.add(self.model_collapsible)

        self.id.spinbox.valueChanged.connect(self.update_neuron_id)

        self.plot = SingleNeuronPlotCollapsible(self, app=app, network=network,
                                                interface=self.neuron_interface)

        self.add(self.plot)

        self.current_control_collapsible = CollapsibleWidget(self, 'current_control')

        self.current_control_frame: Optional[CurrentControlFrame] = None
        self.set_current_control_widget(interface=self.neuron_interface, window=window)

        self.add(self.current_control_collapsible)

    def set_current_control_widget(self, interface: NeuronInterface, window):
        self.current_control_frame = CurrentControlFrame(
            parent=self,
            model=interface.current_injection_function,
            window=window
        )
        self.current_control_collapsible.add(self.current_control_frame)

    def update_neuron_id(self):
        prev_model_id = self.neuron_interface.model_id
        self.neuron_interface.id = self.id.spinbox.value()
        self.model_frame.actualize_values(keys=self.neuron_interface.model_variables)
        if bool(self.neuron_interface.model_id != prev_model_id) is True:
            self.model_frame.reset_preset_combobox()
        self.model_frame.preset_combo_box_frame.combo_box.setCurrentIndex(0)
        self.neuron_interface.u = 0.1
        self.neuron_interface.v = 0

    def update_plots(self, t, t_mod) -> None:
        self.neuron_interface.update_plot(t, t_mod)
        self.plot.canvas.update()

    def set_id(self, neuron_id):
        prev_model_id = self.neuron_interface.model_id
        self.id.spinbox.setValue(neuron_id)

        if self.neuron_interface.model_id != prev_model_id:
            self.model_frame.reset_preset_combobox()

    def set_preset(self, preset: str):
        self.model_frame.preset_combo_box_frame.combo_box.setCurrentText(preset)


class SingleNeuronCollapsibleContainer:

    def __init__(self):

        self.interfaced_neurons_dct = {}
        # self.interfaced_neurons_list: list[SingleNeuronCollapsible] = []
        if not hasattr(self, 'plotting_config'):
            self.plotting_config: Optional[PlottingConfig] = None

    def add_interfaced_neuron(self, network: SpikingNeuralNetwork, app,
                              title=None,
                              neuron_model=MultiModelNeuronStateTensor,
                              neuron_id: Optional[int] = None, preset=None, window=None,
                              ):
        if self.plotting_config is None:
            self.plotting_config = network.plotting_config

        index = len(self.interfaced_neurons_list)

        if title is None:
            title = 'Neuron' + str(index)

        assert title not in self.interfaced_neurons_dct.keys()

        neuron_collapsible = SingleNeuronCollapsible(
            self, network, title=title, window=window, app=app, neuron_model=neuron_model)

        self.interfaced_neurons_dct[title] = neuron_collapsible
        neuron_collapsible.plot.canvas.plot_widget.cam_reset()

        if neuron_id is not None:
            assert self.interfaced_neurons_list[index] == neuron_collapsible
            self.interface_neuron(interfaced_neuron_index=index, neuron_id=neuron_id, preset=preset)

        return neuron_collapsible

    def get_collapsible(self, id_):
        if isinstance(id_, SingleNeuronCollapsible):
            return id_
        elif isinstance(id_, str):
            nc: SingleNeuronCollapsible = self.interfaced_neurons_dct[id_]
        elif isinstance(id_, int):
            nc: SingleNeuronCollapsible = self.interfaced_neurons_list[id_]
        else:
            raise TypeError
        return nc

    def interface_neuron(self, interfaced_neuron_index, neuron_id, preset=None):
        self.interfaced_neurons_list[interfaced_neuron_index].set_id(neuron_id)
        if preset is not None:
            self.set_neuron_model_preset(interfaced_neuron_index=interfaced_neuron_index, preset=preset)

    @property
    def interfaced_neurons_list(self):
        return list(self.interfaced_neurons_dct.values())

    def set_neuron_model_preset(self, interfaced_neuron_index, preset):
        self.interfaced_neurons_list[interfaced_neuron_index].set_preset(preset)

    def unregister_registered_buffers(self):
        for n in self.interfaced_neurons_list:
            n.neuron_interface.plot.unregister_registered_buffers()

    def update_interfaced_neuron_plots(self, t):
        t_mod = t % self.plotting_config.voltage_plot_length
        for n in self.interfaced_neurons_list:
            n.update_plots(t, t_mod)

    def sync_model_variables(self, neuron0, neuron1, bidirectional: bool = True):
        neuron0: SingleNeuronCollapsible = self.get_collapsible(neuron0)
        neuron1: SingleNeuronCollapsible = self.get_collapsible(neuron1)

        neuron0.model_frame.sliders.sync_sliders(neuron1.model_frame.sliders)
        if bidirectional is True:
            neuron1.model_frame.sliders.sync_sliders(neuron0.model_frame.sliders)

    def sync_signal(self, neuron0, neuron1, bidirectional: bool = True):
        neuron0: SingleNeuronCollapsible = self.get_collapsible(neuron0)
        neuron1: SingleNeuronCollapsible = self.get_collapsible(neuron1)

        neuron0.current_control_frame.sliders.sync_sliders(neuron1.current_control_frame.sliders)
        if bidirectional is True:
            neuron1.current_control_frame.sliders.sync_sliders(neuron0.current_control_frame.sliders)
        # neuron1.current_control_frame.sliders.disable()
        # neuron1.neuron_interface.current_injection_function = neuron0.neuron_interface.current_injection_function
