from dataclasses import dataclass
from typing import Union

import torch

from network.network_state.state_tensor import NeuronStateRowCollection, NeuronStateTensor, StateRow
from network.network_structures import NeuronTypes


@dataclass(frozen=True)
class IzhikevichPreset:

    a: float
    b: float
    c: float
    d: float


@dataclass
class IzhikevichPresets:

    RS: IzhikevichPreset = IzhikevichPreset(a=0.02, b=0.2, c=-65., d=8.)
    IB: IzhikevichPreset = IzhikevichPreset(a=0.02, b=0.2, c=-55., d=4.)
    CH: IzhikevichPreset = IzhikevichPreset(a=0.02, b=0.2, c=-50., d=2.)
    FS: IzhikevichPreset = IzhikevichPreset(a=0.1, b=0.2, c=-65., d=2.)
    FS25: IzhikevichPreset = IzhikevichPreset(a=0.09, b=0.24, c=-65., d=2.)
    TC: IzhikevichPreset = IzhikevichPreset(a=0.02, b=0.25, c=-65., d=0.05)
    RZ: IzhikevichPreset = IzhikevichPreset(a=0.1, b=0.26, c=-65., d=2.)
    LTS: IzhikevichPreset = IzhikevichPreset(a=0.02, b=0.25, c=-65., d=2.)

    tonic_spiking: IzhikevichPreset = IzhikevichPreset(a=0.02, b=0.2, c=-65., d=6.)
    phasic_spiking: IzhikevichPreset = IzhikevichPreset(a=0.02, b=0.25, c=-65., d=6.)
    tonic_bursting: IzhikevichPreset = IzhikevichPreset(a=0.02, b=0.2, c=-50., d=2.)
    phasic_bursting: IzhikevichPreset = IzhikevichPreset(a=0.02, b=0.25, c=-55., d=0.05)
    mixed_mode: IzhikevichPreset = IzhikevichPreset(a=0.02, b=0.2, c=-55., d=4)
    spike_frequency_adaptation: IzhikevichPreset = IzhikevichPreset(a=0.01, b=0.2, c=-65., d=8)
    class_1_exc: IzhikevichPreset = IzhikevichPreset(a=0.02, b=-0.1, c=-65., d=6)
    class_2_exc: IzhikevichPreset = IzhikevichPreset(a=0.2, b=0.26, c=-65., d=0)
    spike_latency: IzhikevichPreset = IzhikevichPreset(a=0.02, b=0.2, c=-65., d=6.)
    subthreshold_oscillations: IzhikevichPreset = IzhikevichPreset(a=0.05, b=0.26, c=-60., d=0.)
    resonator: IzhikevichPreset = IzhikevichPreset(a=0.1, b=0.26, c=-60., d=-1.)
    integrator: IzhikevichPreset = IzhikevichPreset(a=0.02, b=-0.1, c=-55., d=6)
    rebound_spike: IzhikevichPreset = IzhikevichPreset(a=0.03, b=0.25, c=-60., d=4)
    rebound_burst: IzhikevichPreset = IzhikevichPreset(a=0.03, b=0.25, c=-52., d=0)
    threshold_variability: IzhikevichPreset = IzhikevichPreset(a=0.03, b=0.25, c=-60., d=4)
    bistability: IzhikevichPreset = IzhikevichPreset(a=1, b=1.5, c=-60., d=0)
    depolarizing_after_potential: IzhikevichPreset = IzhikevichPreset(a=1, b=.2, c=-60., d=-21)
    accommodation: IzhikevichPreset = IzhikevichPreset(a=0.02, b=1, c=-55., d=4)
    inh_induced_spiking: IzhikevichPreset = IzhikevichPreset(a=-0.02, b=-1, c=-60., d=8)
    inh_induced_bursting: IzhikevichPreset = IzhikevichPreset(a=-0.026, b=-1, c=-45., d=0)


class IzhikevichInitializer:

    def __init__(self, n_neurons, device, mask_inh, mask_exc):

        r = torch.rand(n_neurons, dtype=torch.float32, device=device)
        self.v = -65.
        self.a = .02 + .08 * r * mask_inh
        self.b = .2 + .05 * (1. - r) * mask_inh
        self.c = -65 + 15 * (r ** 2) * mask_exc
        self.d = 2 * mask_inh + (8 - 6 * (r ** 2)) * mask_exc
        self.u = self.b * self.v
        self.v_prev = self.v


class IzhikevichModel(NeuronStateTensor):

    """
    From Simple Model of Spiking Neurons (2003), Eugene M. Izhikevich:

    1. The parameter a describes the timescale of the recovery variable u.
    Smaller values result in slower recovery. A typical value is
    a = 0.02

    2. The parameter b describes the sensitivity of the recovery variable
    u to the subthreshold fluctuations of the membrane potential v.
    Greater values couple v and u more strongly resulting in possible
    subthreshold oscillations and low-threshold spiking dynamics. A
    typical value is b = 0:2. The case b<a(b>a) corresponds
    to saddle-node (Andronov–Hopf) bifurcation of the resting state

    3. The parameter c describes the after-spike reset value of the membrane
    potential v caused by the fast high-threshold K+ conductances. A typical value is
    c = -65 mV.

    4. The parameter d describes after-spike reset of the recovery variable
    u caused by slow high-threshold Na+ and K+ conductances.
    A typical value is d = 2.

    """

    @dataclass(frozen=True)
    class Rows(NeuronStateRowCollection):

        pt: StateRow = StateRow(0, [0, 1], 0.01)
        u: StateRow = StateRow(1)
        v: StateRow = StateRow(2)
        a: StateRow = StateRow(3, [-1, 1], 0.001)
        b: StateRow = StateRow(4, [-2, 5], 0.01)
        c: StateRow = StateRow(5, [-65, -40], 0.1)
        d: StateRow = StateRow(6, [-21, 10], 0.1)
        i: StateRow = StateRow(7)
        i_prev: StateRow = StateRow(8)
        v_prev: StateRow = StateRow(9)

    def __init__(self, n_neurons, device, neuron_types_flags, tensor=None, model_mask=None):
        self._rows = self.Rows()
        self._N = n_neurons

        super().__init__(
            shape=(len(self._rows), n_neurons), dtype=torch.float32,
            neuron_types_flags=neuron_types_flags,
            tensor=tensor, model_mask=model_mask, device=device, presets=IzhikevichPresets(),
            preset_model=IzhikevichPreset
        )

        assert self._model_specific_variables == ['a', 'b', 'c', 'd']

    def default_preset(self, mask_inh, mask_exc, n_neurons=None):
        return IzhikevichInitializer(n_neurons=self._N if n_neurons is None else n_neurons,
                                     device=self._cuda_device,
                                     mask_inh=mask_inh, mask_exc=mask_exc)

    def set_tensor(self, neuron_types_flags):

        mask_inh = neuron_types_flags == NeuronTypes.INHIBITORY.value
        mask_exc = neuron_types_flags == NeuronTypes.EXCITATORY.value

        if self._model_mask is not None:
            mask_inh = mask_inh & self._model_mask
            mask_exc = mask_exc & self._model_mask

        init_values = IzhikevichInitializer(n_neurons=self._N, device=self._cuda_device,
                                            mask_inh=mask_inh, mask_exc=mask_exc)

        self.pt = torch.rand(self._N, dtype=torch.float32, device=self._cuda_device)
        self.v = init_values.v
        self.a = init_values.a
        self.b = init_values.b
        self.c = init_values.c
        self.d = init_values.d
        self.u = init_values.u

        # self.selected = torch.zeros(self._N, dtype=torch.int32, device=device)

    @property
    def model_specific_variables(self):
        return self._model_specific_variables

    def use_preset(self, preset: Union[IzhikevichPreset, str], mask=None):

        preset, mask = self._interpret_use_preset_parameters(preset, mask)

        if self._model_mask is None:
            self.a[mask] = preset.a
            self.b[mask] = preset.b
            self.c[mask] = preset.c
            self.d[mask] = preset.d
        else:
            self._tensor[self._rows.a.index, mask] = preset.a
            self._tensor[self._rows.b.index, mask] = preset.b
            self._tensor[self._rows.c.index, mask] = preset.c
            self._tensor[self._rows.d.index, mask] = preset.d

    @property
    def a(self):
        if self._model_mask is None:
            return self._tensor[self._rows.a.index, :]
        else:
            return self._tensor[self._rows.a.index, self._model_mask]

    @a.setter
    def a(self, v):
        if self._model_mask is None:
            self._tensor[self._rows.a.index, :] = v
        else:
            self._tensor[self._rows.a.index, self._model_mask] = v

    @property
    def b(self):
        if self._model_mask is None:
            return self._tensor[self._rows.b.index, :]
        else:
            return self._tensor[self._rows.b.index, self._model_mask]

    @b.setter
    def b(self, v):
        if self._model_mask is None:
            self._tensor[self._rows.b.index, :] = v
        else:
            self._tensor[self._rows.b.index, self._model_mask] = v

    @property
    def c(self):
        if self._model_mask is None:
            return self._tensor[self._rows.c.index, :]
        else:
            return self._tensor[self._rows.c.index, self._model_mask]

    @c.setter
    def c(self, v):
        if self._model_mask is None:
            self._tensor[self._rows.c.index, :] = v
        else:
            self._tensor[self._rows.c.index, self._model_mask] = v

    @property
    def d(self):
        if self._model_mask is None:
            return self._tensor[self._rows.d.index, :]
        else:
            return self._tensor[self._rows.d.index, self._model_mask]

    @d.setter
    def d(self, v):
        if self._model_mask is None:
            self._tensor[self._rows.d.index, :] = v
        else:
            self._tensor[self._rows.d.index, self._model_mask] = v

