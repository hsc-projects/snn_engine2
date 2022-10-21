from copy import copy, deepcopy
from dataclasses import asdict
import torch
from typing import Callable, Optional, Union

from signaling.signaling import (
    SignalModel)

from network.gpu.visualized_elements.plots import SingleNeuronPlot
from network.spiking_neural_network import SpikingNeuralNetwork
from engine.content.plot_widgets import SingleNeuronPlotWidget
from network.network_structures import ModelIDs, NeuronTypes


class NeuronInterface:

    def __init__(self, neuron_id, network: SpikingNeuralNetwork,
                 plot_widget: Optional[SingleNeuronPlotWidget] = None):

        self._id = neuron_id
        self._model_id = None
        self.network = network

        self.plot = SingleNeuronPlot(self.plot_length)

        self.plot_widget = plot_widget

        self.max_v = 0
        self.max_prev_i = 0

        self.first_plot_run = True

        self.b_current_injection = False
        self.current_injection_function: Optional[Union[SignalModel, Callable]] = None

        self.current_scale_reset_threshold_up = 0.8
        self.current_scale_reset_threshold_down = 0
        self.voltage_scale_reset_threshold_up = 5
        self.voltage_scale_reset_threshold_down = 0

        self.preset_dct = self._get_presets()
        self.preset_dct['custom'] = copy(self.preset_dct['initial'])

        # self.network.GPU.N_states.preset_model(**self.preset_dct['initial'])

        self.set_current_injection(activate=True, mode='step_signal')
        if self.model_id == ModelIDs.rulkov.value:
            self.current_injection_function.amplitude = 0
        self.plotting_u_factor = 1

        # self.custom_presets = SortedDict()

    def _get_presets(self) -> dict:
        preset_dct = {'initial': None}
        preset_dct.update(asdict(self.presets))
        return preset_dct

    def reset_special_presets(self, b_neuron_model_changed):
        if b_neuron_model_changed is True:
            self.preset_dct['custom'] = {}
            self.current_model_config = self.get_model_class_instance().default_preset(
                n_neurons=1,
                mask_inh=self.type == NeuronTypes.INHIBITORY.value,
                mask_exc=self.type == NeuronTypes.EXCITATORY.value)
        self.preset_dct['initial'] = self.current_model_config
        if b_neuron_model_changed is False:
            self.preset_dct['custom'] = copy(self.preset_dct['initial'])

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, i):
        previous_model_id = self.model_id
        self._id = i
        self._model_id = self.network.neurons.N_flags.model[self.id]
        # print('self._model_id =', self._model_id)
        self.set_plotting_u_factor()
        b_neuron_model_changed = bool(self.model_id != previous_model_id)

        if b_neuron_model_changed is True:
            self.preset_dct = self._get_presets()
        self.reset_special_presets(b_neuron_model_changed)

    @property
    def current_model_config(self):
        model = {}
        for k in self.model_variables:
            model[k] = getattr(self, k).item()
        return model

    @current_model_config.setter
    def current_model_config(self, value):
        for k in self.model_variables:
            setattr(self, k, getattr(value, k))
        self.v = value.v
        self.u = value.u
        self.v_prev = value.v_prev

    def use_preset(self, key):
        # print(self.preset_dct[key])
        for k, v in self.preset_dct[key].items():
            setattr(self, k, v)

    @property
    def presets(self):
        # return self.network.GPU.N_states.presets
        return self.network.neurons.N_states.get_model_class_instance(self.model_id).presets

    @property
    def group(self):
        # return self.network.GPU.N_G[self.id, self.network.network_config.N_G_group_id_col]
        return self.network.neurons.N_flags.group[self.id]

    @property
    def type(self):
        # return self.network.GPU.N_G[self.id, self.network.network_config.N_G_neuron_type_col]
        return self.network.neurons.N_flags.type[self.id]

    def get_model_class_instance(self):
        return self.network.neurons.N_states.get_model_class_instance(self.model_id)

    @property
    def model_id(self):
        if self._model_id is None:
            self._model_id = self.network.neurons.N_flags.model[self.id]
        return self._model_id

    @property
    def model_variables(self):
        return self.get_model_class_instance()._model_specific_variables

    def register_vbos(self):
        self.plot.init_cuda_attributes(self.network.simulation_gpu.device)
        self.plot.line.colors_gpu.tensor[:, 3] = 0

    def link_plot_widget(self, plot_widget: SingleNeuronPlotWidget):
        self.plot_widget = plot_widget
        self.plot_widget.view.add(self.plot)

    @property
    def pt(self):
        return self.network.neurons.N_states.pt[self.id]

    @pt.setter
    def pt(self, v):
        self.network.neurons.N_states.pt[self.id] = v

    @property
    def v(self):
        return self.network.neurons.N_states.v[self.id]

    @v.setter
    def v(self, value):
        self.network.neurons.N_states.v[self.id] = value

    @property
    def u(self):
        return self.network.neurons.N_states.u[self.id]

    @u.setter
    def u(self, v):
        self.network.neurons.N_states.u[self.id] = v

    @property
    def i(self):
        return self.network.neurons.N_states.i[self.id]

    @i.setter
    def i(self, v):
        self.network.neurons.N_states.i[self.id] = v

    def set_current_injection(self, activate: bool, mode):

        self.b_current_injection = activate
        self.current_injection_function: SignalModel = deepcopy(getattr(self.network.signal_collection, mode))

    @property
    def i_prev(self):
        return self.network.neurons.N_states.i_prev[self.id]

    @i_prev.setter
    def i_prev(self, v):
        self.network.neurons.N_states.i_prev[self.id] = v

    @property
    def v_prev(self):
        return self.network.neurons.N_states.v_prev[self.id]

    @v_prev.setter
    def v_prev(self, v):
        self.network.neurons.N_states.v_prev[self.id] = v

    @property
    def plot_length(self):
        return self.network.plotting_config.voltage_plot_length

    def _rescale_plot(self):

        if self.first_plot_run is True:
            self.first_plot_run = False

        max_v = torch.max(self.plot.line.pos_gpu.tensor[0: self.plot_length, 1]).item()
        if max_v > self.voltage_scale_reset_threshold_up:
            self.plot_widget.y_axis_right.scale *= (max_v / 2)

        max_i = torch.max(self.plot.line.pos_gpu.tensor[self.plot_length: 2 * self.plot_length, 1]).item()
        if max_i > self.current_scale_reset_threshold_up:
            self.plot_widget.y_axis.scale *= (max_i + 1)

        if (max_v > self.voltage_scale_reset_threshold_up) or (max_i > self.current_scale_reset_threshold_up):
            self.plot_widget.cam_reset()

    def set_plotting_u_factor(self):
        if self.model_id == ModelIDs.rulkov.value:
            self.plotting_u_factor = 500
        else:
            self.plotting_u_factor = 1

    def update_plot(self, t, t_mod):

        if t_mod == 0:
            self.plot.line.colors_gpu.tensor[:, 3] = 0

        if (t_mod == (self.plot_length - 1)) is True:
            self._rescale_plot()

        # print(self.v)

        self.plot.line.pos_gpu.tensor[t_mod, 1] = self.v / self.plot_widget.y_axis_right.scale
        self.plot.line.pos_gpu.tensor[t_mod + self.plot_length, 1] = \
            self.i_prev / self.plot_widget.y_axis.scale
        # print(self.plotting_u_factor, self.u)
        self.plot.line.pos_gpu.tensor[t_mod + 2 * self.plot_length, 1] = \
            self.plotting_u_factor * (self.u / self.plot_widget.y_axis_right.scale)

        # print(self.id, self.u)

        if self.b_current_injection is True:
            self.i += self.current_injection_function(t, t_mod)

        if t_mod > 0:
            self.plot.line.colors_gpu.tensor[t_mod, 3] = 1
            self.plot.line.colors_gpu.tensor[t_mod + self.plot_length, 3] = 1
            self.plot.line.colors_gpu.tensor[t_mod + 2 * self.plot_length, 3] = 1

    @property
    def state_tensor(self):
        return self.network.neurons.N_states

# class IzhikevichNeuronsInterface(NeuronInterface):
#
#     def __init__(self, neuron_id, network: SpikingNeuronNetwork,
#                  plot_widget: Optional[SingleNeuronPlotWidget] = None):
#         super().__init__(neuron_id, network, plot_widget=plot_widget)

    @property
    def a(self):
        return self.state_tensor.a[self.id]

    @a.setter
    def a(self, v):
        assert self.model_id == ModelIDs.izhikevich.value
        self.preset_dct['custom']['a'] = v
        # print(self.network.GPU.N_states.tensor[:, self.id])
        self.state_tensor.a[self.id] = v

    @property
    def b(self):
        return self.state_tensor.b[self.id]

    @b.setter
    def b(self, v):
        assert self.model_id == ModelIDs.izhikevich.value
        self.preset_dct['custom']['b'] = v
        self.state_tensor.b[self.id] = v

    @property
    def c(self):
        return self.state_tensor.c[self.id]

    @c.setter
    def c(self, v):
        assert self.model_id == ModelIDs.izhikevich.value
        self.preset_dct['custom']['c'] = v
        self.state_tensor.c[self.id] = v

    @property
    def d(self):
        return self.state_tensor.d[self.id]

    @d.setter
    def d(self, v):
        assert self.model_id == ModelIDs.izhikevich.value
        self.preset_dct['custom']['d'] = v
        self.state_tensor.d[self.id] = v

    @property
    def theta(self):
        # assert self.model_id == ModelIDs.rulkov.value
        return self.state_tensor.theta[self.id]

    @theta.setter
    def theta(self, v):
        assert self.model_id == ModelIDs.rulkov.value
        self.preset_dct['custom']['theta'] = v
        self.state_tensor.theta[self.id] = v

    @property
    def kappa(self):
        # assert self.model_id == ModelIDs.rulkov.value
        return self.state_tensor.kappa[self.id]

    @kappa.setter
    def kappa(self, v):
        assert self.model_id == ModelIDs.rulkov.value
        self.preset_dct['custom']['kappa'] = v
        self.state_tensor.kappa[self.id] = v

    @property
    def epsilon(self):
        # assert self.model_id == ModelIDs.rulkov.value
        return self.state_tensor.epsilon[self.id]

    @epsilon.setter
    def epsilon(self, v):
        assert self.model_id == ModelIDs.rulkov.value
        self.preset_dct['custom']['epsilon'] = v
        self.state_tensor.epsilon[self.id] = v

    @property
    def gamma(self):
        # assert self.model_id == ModelIDs.rulkov.value
        return self.state_tensor.gamma[self.id]

    @gamma.setter
    def gamma(self, v):
        assert self.model_id == ModelIDs.rulkov.value
        self.preset_dct['custom']['gamma'] = v
        self.state_tensor.gamma[self.id] = v
