from dataclasses import dataclass

import torch

from .state_tensor import StateRow, StateTensor
from network.gpu.visualized_elements.neuron_visual import NeuronVisual
import pandas as pd


@dataclass(frozen=True)
class FlagRow(StateRow):
    step_size: int = 1


class NeuronFlags(StateTensor):

    @dataclass(frozen=True)
    class Rows:
        b_sensory_input: StateRow = FlagRow(0, [0, 1])
        type: StateRow = FlagRow(1, [0, 1])
        group: StateRow = FlagRow(2)
        model: StateRow = FlagRow(3, [0, 1])
        selected: StateRow = FlagRow(4, [0, 1])

        def __len__(self):
            return 5

    def __init__(self, n_neurons, device):
        self._rows = self.Rows()
        super().__init__(shape=(len(self._rows), n_neurons), device=device, dtype=torch.int32)
        self.id = torch.arange(n_neurons, device=self._cuda_device)

    @property
    def b_sensory_input(self):
        return self._tensor[self._rows.b_sensory_input.index, :]

    @b_sensory_input.setter
    def b_sensory_input(self, v):
        self._tensor[self._rows.b_sensory_input.index, :] = v

    @property
    def type(self):
        return self._tensor[self._rows.type.index, :]

    @type.setter
    def type(self, v):
        self._tensor[self._rows.type.index, :] = v

    @property
    def group(self):
        return self._tensor[self._rows.group.index, :]

    @group.setter
    def group(self, v):
        self._tensor[self._rows.group.index, :] = v

    @property
    def model(self):
        return self._tensor[self._rows.model.index, :]

    @model.setter
    def model(self, v):
        self._tensor[self._rows.model.index, :] = v

    @property
    def selected(self):
        return self._tensor[self._rows.selected.index, :]

    @selected.setter
    def selected(self, v):
        self._tensor[self._rows.selected.index, :] = v

    def select_by_groups(self, groups, ntype=None):
        self.selected[:] = 0
        for g in groups:
            if ntype is None:
                self.selected += (self.group == g)
            else:
                self.selected += (self.group == g) & (self.type == ntype)
        # self.selected = self.selected > 0
        return self.selected > 0

    def select_ids_by_groups(self, groups, ntype=None):
        return self.id[self.select_by_groups(groups, ntype=ntype)]

    def validate(self, neurons: NeuronVisual, N_pos):
        if (self.type == 0).sum() > 0:
            raise AssertionError

        cond0 = (self.type[:-1]
                 .masked_select(self.type.diff() < 0).size(dim=0) > 0)
        cond1 = (self.group[:-1]
                 .masked_select(self.group.diff() < 0).size(dim=0) != 1)
        if cond0 or cond1:

            idcs1 = (self.group.diff() < 0).nonzero()
            df = pd.DataFrame(N_pos.tensor[:, :3].cpu().numpy())
            df[['0g', '1g', '2g']] = neurons.shape
            df['group'] = self.group.cpu().numpy()
            # print(G_pos)
            print(df)
            df10 = df.iloc[int(idcs1[0]) - 2: int(idcs1[0]) + 3, :]
            df11 = df.iloc[int(idcs1[-1]) - 2: int(idcs1[-1]) + 3, :]
            print(df10)
            print(df11)
            raise AssertionError