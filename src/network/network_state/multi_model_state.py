from dataclasses import dataclass
import torch
from typing import Union

from .izhikevich_model import IzhikevichModel
from .rulkov_model import RulkovModel
from .state_tensor import StateTensor, StateRow
from .neuron_flags import NeuronFlags

from network.network_structures import ModelIDs


class MultiModelNeuronStateTensor(StateTensor):
    @dataclass(frozen=True)
    class Rows:

        pt: StateRow = StateRow(0, [0, 1], 0.01)
        u: StateRow = StateRow(1)
        v: StateRow = StateRow(2)
        a: StateRow = StateRow(3, [-1, 1], 0.001)
        b: StateRow = StateRow(4, [-2, 5], 0.01)
        c: StateRow = StateRow(5, [-65, -40], 0.1)
        d: StateRow = StateRow(6, [-21, 10], 0.1)
        theta: StateRow = StateRow(3, [0, 5], 0.001)
        kappa: StateRow = StateRow(4, [-10, 10], 0.01)
        epsilon: StateRow = StateRow(5, [0, 2], 0.001)
        gamma: StateRow = StateRow(6, [0, 50], 0.1)
        i: StateRow = StateRow(7)
        i_prev: StateRow = StateRow(8)
        v_prev: StateRow = StateRow(9)

    def __len__(self):
        return len(RulkovModel.Rows())

    def __init__(self, n_neurons, device, flag_tensor: NeuronFlags):

        super().__init__(shape=(len(self), n_neurons), dtype=torch.float32, device=device)

        self.flag_tensor = flag_tensor

        izhikevich_mask = (flag_tensor.model == ModelIDs.izhikevich.value)
        self.izhikevich_neurons = IzhikevichModel(n_neurons=len(flag_tensor.model[izhikevich_mask]),
                                                  device=device, neuron_types_flags=flag_tensor.type,
                                                  tensor=self.tensor, model_mask=izhikevich_mask)

        rulkov_mask = (flag_tensor.model == ModelIDs.rulkov.value)
        self.rulkov_neurons = RulkovModel(n_neurons=len(flag_tensor.model[rulkov_mask]),
                                          device=device, neuron_types_flags=flag_tensor.type,
                                          tensor=self.tensor, model_mask=rulkov_mask)

        self.model_instances: list[Union[IzhikevichModel, RulkovModel]] = [self.izhikevich_neurons, self.rulkov_neurons]

    def get_model_class_instance(self, model_id) -> Union[IzhikevichModel, RulkovModel]:
        if model_id == ModelIDs.izhikevich.value:
            return self.izhikevich_neurons
        elif model_id == ModelIDs.rulkov.value:
            return self.rulkov_neurons
        else:
            raise ValueError(f"model id = {model_id}")

    def change_model(self, neuron_ids, model_id):

        target_model = self.get_model_class_instance(model_id=model_id)

        target_model._model_mask[neuron_ids] = True

        for other_model in self.model_instances:
            if other_model is not target_model:
                other_model._model_mask[neuron_ids] = False

    @property
    def pt(self):
        return self._tensor[self._rows.pt.index, :]

    @pt.setter
    def pt(self, v):
        self._tensor[self._rows.pt.index, :] = v

    @property
    def u(self):
        return self._tensor[self._rows.u.index, :]

    @u.setter
    def u(self, v):
        self._tensor[self._rows.u.index, :] = v

    @property
    def v(self):
        return self._tensor[self._rows.v.index, :]

    @v.setter
    def v(self, v):
        self._tensor[self._rows.v.index, :] = v

    @property
    def a(self):
        return self._tensor[self._rows.a.index, :]

    @a.setter
    def a(self, v):
        self._tensor[self._rows.a.index, :] = v

    @property
    def b(self):
        return self._tensor[self._rows.b.index, :]

    @b.setter
    def b(self, v):
        self._tensor[self._rows.b.index, :] = v

    @property
    def c(self):
        return self._tensor[self._rows.c.index, :]

    @c.setter
    def c(self, v):
        self._tensor[self._rows.c.index, :] = v

    @property
    def d(self):
        return self._tensor[self._rows.d.index, :]

    @d.setter
    def d(self, v):
        self._tensor[self._rows.d.index, :] = v

    @property
    def i(self):
        return self._tensor[self._rows.i.index, :]

    @i.setter
    def i(self, v):
        self._tensor[self._rows.i.index, :] = v

    @property
    def i_prev(self):
        return self._tensor[self._rows.i_prev.index, :]

    @i_prev.setter
    def i_prev(self, v):
        self._tensor[self._rows.i_prev.index, :] = v

    @property
    def v_prev(self):
        return self._tensor[self._rows.v_prev.index, :]

    @i_prev.setter
    def v_prev(self, v):
        self._tensor[self._rows.v_prev.index, :] = v

    @property
    def theta(self):
        return self._tensor[self._rows.theta.index, :]

    @theta.setter
    def theta(self, v):
        self._tensor[self._rows.theta.index, :] = v

    @property
    def kappa(self):
        return self._tensor[self._rows.kappa.index, :]

    @kappa.setter
    def kappa(self, v):
        self._tensor[self._rows.kappa.index, :] = v

    @property
    def epsilon(self):
        return self._tensor[self._rows.epsilon.index, :]

    @epsilon.setter
    def epsilon(self, v):
        self._tensor[self._rows.epsilon.index, :] = v

    @property
    def gamma(self):
        return self._tensor[self._rows.gamma.index, :]

    @gamma.setter
    def gamma(self, v):
        self._tensor[self._rows.gamma.index, :] = v
