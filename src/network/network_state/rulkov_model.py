from dataclasses import dataclass
from typing import Union

import torch

from network.network_state.state_tensor import NeuronStateRowCollection, NeuronStateTensor, StateRow
from network.network_structures import NeuronTypes


@dataclass(frozen=True)
class RulkovPreset:

    theta: float
    kappa: float
    epsilon: float
    gamma: float


@dataclass
class RulkovPresets:

    tonic_spiking: RulkovPreset = RulkovPreset(theta=0.1, kappa=.5, epsilon=.5, gamma=.5)
    spike_frequency_adaptation: RulkovPreset = RulkovPreset(
        theta=.1, kappa=1., epsilon=.001, gamma=5.)
    rebound_spiking: RulkovPreset = RulkovPreset(theta=.02, kappa=2., epsilon=.01, gamma=.2)
    accommodation: RulkovPreset = RulkovPreset(theta=.12, kappa=3., epsilon=.02, gamma=.4)
    spike_latency: RulkovPreset = RulkovPreset(theta=.1, kappa=0., epsilon=.005, gamma=.4)
    inh_induced_spiking: RulkovPreset = RulkovPreset(theta=.02, kappa=-1, epsilon=.002, gamma=.4)

    # harmonic input i(t) = phi * cos((omega * pi * t / 1000) + nu)
    low_pass_filter: RulkovPreset = RulkovPreset(theta=1 / 7, kappa=.1, epsilon=.005, gamma=2)
    low_pass_filter2: RulkovPreset = RulkovPreset(theta=.3, kappa=.18, epsilon=.24, gamma=.1)
    high_pass_filter: RulkovPreset = RulkovPreset(theta=1 / 7, kappa=2., epsilon=.005, gamma=2)
    high_pass_filter2: RulkovPreset = RulkovPreset(theta=1, kappa=2.3, epsilon=.091, gamma=2.8)


class RulkovInitializer:

    def __init__(self, n_neurons=None, device=None, mask_inh=None, mask_exc=None):

        tonic_spiking_preset = RulkovPreset(theta=0.1, kappa=.5, epsilon=.5, gamma=.5)
        # r = torch.rand(n_neurons, dtype=torch.float32, device=device)
        self.v = -65.
        self.theta = tonic_spiking_preset.theta
        self.kappa = tonic_spiking_preset.kappa
        self.epsilon = tonic_spiking_preset.epsilon
        self.gamma = tonic_spiking_preset.gamma
        self.u = 0.01
        self.v_prev = -65.


class RulkovModel(NeuronStateTensor):

    @dataclass(frozen=True)
    class Rows(NeuronStateRowCollection):

        pt: StateRow = StateRow(0, [0, 1], 0.01)
        u: StateRow = StateRow(1)
        v: StateRow = StateRow(2)
        theta: StateRow = StateRow(3, [0, 1], 0.001)
        kappa: StateRow = StateRow(4, [-5, 5], 0.01)
        epsilon: StateRow = StateRow(5, [0, 1], 0.001)
        gamma: StateRow = StateRow(6, [0, 50], 0.1)
        i: StateRow = StateRow(7)
        i_prev: StateRow = StateRow(8)
        v_prev: StateRow = StateRow(9)

    def __init__(self, n_neurons, device, neuron_types_flags, tensor=None, model_mask=None):
        self._rows = self.Rows()
        self._N = n_neurons
        super().__init__(shape=(len(self._rows), n_neurons), dtype=torch.float32,
                         neuron_types_flags=neuron_types_flags,
                         tensor=tensor, model_mask=model_mask, device=device,
                         presets=RulkovPresets(), preset_model=RulkovPreset)

        assert self._model_specific_variables == ['theta', 'kappa', 'epsilon', 'gamma']

    def set_tensor(self, neuron_types_flags):
        pass

    def default_preset(self, mask_inh=None, mask_exc=None, n_neurons=None):
        return RulkovInitializer(n_neurons=self._N if n_neurons is None else n_neurons,
                                 device=self._cuda_device, mask_inh=mask_inh, mask_exc=mask_exc)

    def use_preset(self, preset: Union[RulkovPreset, str], mask=None):

        preset, mask = self._interpret_use_preset_parameters(preset, mask)

        self.theta[mask] = preset.theta
        self.kappa[mask] = preset.kappa
        self.epsilon[mask] = preset.epsilon
        self.gamma[mask] = preset.gamma

    @property
    def theta(self):
        if self._model_mask is None:
            return self._tensor[self._rows.theta.index, :]
        else:
            return self._tensor[self._rows.theta.index, self._model_mask]

    @theta.setter
    def theta(self, v):
        if self._model_mask is None:
            self._tensor[self._rows.theta.index, :] = v
        else:
            self._tensor[self._rows.theta.index, self._model_mask] = v

    @property
    def kappa(self):
        if self._model_mask is None:
            return self._tensor[self._rows.kappa.index, :]
        else:
            return self._tensor[self._rows.kappa.index, self._model_mask]

    @kappa.setter
    def kappa(self, v):
        if self._model_mask is None:
            self._tensor[self._rows.kappa.index, :] = v
        else:
            self._tensor[self._rows.kappa.index, self._model_mask] = v

    @property
    def epsilon(self):
        if self._model_mask is None:
            return self._tensor[self._rows.epsilon.index, :]
        else:
            return self._tensor[self._rows.epsilon.index, self._model_mask]

    @epsilon.setter
    def epsilon(self, v):
        if self._model_mask is None:
            self._tensor[self._rows.epsilon.index, :] = v
        else:
            self._tensor[self._rows.epsilon.index, self._model_mask] = v

    @property
    def gamma(self):
        if self._model_mask is None:
            return self._tensor[self._rows.gamma.index, :]
        else:
            return self._tensor[self._rows.gamma.index, self._model_mask]

    @gamma.setter
    def gamma(self, v):
        if self._model_mask is None:
            self._tensor[self._rows.gamma.index, :] = v
        else:
            self._tensor[self._rows.gamma.index, self._model_mask] = v
