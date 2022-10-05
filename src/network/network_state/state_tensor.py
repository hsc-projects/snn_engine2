from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd
import torch
from typing import Optional, Union


@dataclass(frozen=True)
class StateRow:

    index: int
    interval: Optional[Union[list, pd.Interval]] = None
    step_size: Optional[Union[int, float]] = None


@dataclass(frozen=True)
class NeuronStateRowCollection:

    pt: StateRow = StateRow(0, [0, 1])
    v: StateRow = StateRow(1)
    u: StateRow = StateRow(2)
    i: StateRow = StateRow(3)
    i_prev: StateRow = StateRow(4)
    v_prev: StateRow = StateRow(5)

    def __post_init__(self):
        keys = list(asdict(self).keys())
        vs = np.array([x['index'] for x in list(asdict(self).values())])

        if not np.max(vs) + 1 == len(self):
            raise AttributeError

        if not np.math.factorial(getattr(self,  keys[np.argmax(vs)]).index) == np.cumprod(vs[vs > 0])[-1]:
            raise AttributeError

    def __len__(self):
        return self.v_prev.index + 1


class Sliders:
    def __init__(self, rows):
        for k in asdict(rows):
            setattr(self, k, None)


class StateTensor:

    """
    Access tensor rows via class-properties for readability and
    where performance is not crucial."""

    class Rows:
        def __len__(self):
            pass

    _rows = None

    def __init__(self, shape, device, tensor: Optional[torch.Tensor] = None, dtype=None, model_mask=None):
        self._cuda_device = device
        if (not hasattr(self, '_rows')) or (self._rows is None):
            self._rows = self.Rows()
        self._model_mask = model_mask
        self._tensor: Optional[torch.Tensor] = None
        if tensor is not None:
            self.tensor: Optional[torch.Tensor] = tensor
        else:
            self.tensor = torch.zeros(shape, dtype=dtype, device=self._cuda_device )

    @classmethod
    def __len__(cls):
        return len(cls.Rows())

    def __str__(self):
        return f"{self.__class__.__name__}:\n" + str(self.tensor)

    # noinspection PyPropertyDefinition
    @classmethod
    @property
    def rows(cls):
        if cls._rows is None:
            cls._rows = cls.Rows()
        return cls._rows

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor

    @tensor.setter
    def tensor(self, v):
        if self._tensor is None:
            if ((not len(v.shape) == 2)
                    or (not v.shape[0] >= len(self))):
                raise ValueError('tensor.shape[0] < len(self))')

            self._tensor: torch.Tensor = v
            self.has_tensor = True
        else:
            raise AttributeError

    def data_ptr(self):
        return self._tensor.data_ptr()

    @property
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._tensor.cpu().numpy())

    def set_tensor(self, **kwargs):
        pass


class NeuronStateTensor(StateTensor):

    @dataclass(frozen=True)
    class Rows(NeuronStateRowCollection):
        pass

    def __init__(self, shape, dtype, neuron_types_flags,
                 presets, preset_model,
                 device,
                 tensor=None, model_mask=None,
                 call_set_tensor=True):
        # self._rows = self.Rows
        super().__init__(shape, dtype=dtype, tensor=tensor, model_mask=model_mask, device=device, )
        self.selected = None
        self.presets = presets
        self.preset_model = preset_model
        if call_set_tensor is True:
            self.set_tensor(neuron_types_flags=neuron_types_flags)
        if (not hasattr(self, '_model_specific_variables')) or (self._model_specific_variables is None):
            current_vars = list(asdict(self._rows).keys())
            common_vars = list(asdict(NeuronStateRowCollection()).keys())
            self._model_specific_variables = [k for k in current_vars if k not in common_vars]

    def set_tensor(self, neuron_types_flags):
        pass

    def _interpret_use_preset_parameters(self, preset, mask=None):

        if isinstance(preset, str):
            preset = getattr(self.presets, preset)

        if mask is None:
            mask = self.selected

        if self._model_mask is not None:
            if mask is not None:
                mask = mask & self._model_mask
            else:
                mask = self._model_mask

        return preset, mask

    @property
    def pt(self):
        if self._model_mask is None:
            return self._tensor[self._rows.pt.index, :]
        else:
            return self._tensor[self._rows.pt.index, self._model_mask]

    @pt.setter
    def pt(self, v):
        if self._model_mask is None:
            self._tensor[self._rows.pt.index, :] = v
        else:
            self._tensor[self._rows.pt.index, self._model_mask] = v

    @property
    def u(self):
        if self._model_mask is None:
            return self._tensor[self._rows.u.index, :]
        else:
            return self._tensor[self._rows.u.index, self._model_mask]

    @u.setter
    def u(self, v):
        if self._model_mask is None:
            self._tensor[self._rows.u.index, :] = v
        else:
            self._tensor[self._rows.u.index, self._model_mask] = v

    @property
    def v(self):
        if self._model_mask is None:
            return self._tensor[self._rows.v.index, :]
        else:
            return self._tensor[self._rows.v.index, self._model_mask]

    @v.setter
    def v(self, v):
        if self._model_mask is None:
            self._tensor[self._rows.v.index, :] = v
        else:
            self._tensor[self._rows.v.index, self._model_mask] = v

    @property
    def i(self):
        if self._model_mask is None:
            return self._tensor[self._rows.i.index, :]
        else:
            return self._tensor[self._rows.i.index, self._model_mask]

    @i.setter
    def i(self, v):
        if self._model_mask is None:
            self._tensor[self._rows.i.index, :] = v
        else:
            self._tensor[self._rows.i.index, self._model_mask] = v

    @property
    def i_prev(self):
        if self._model_mask is None:
            return self._tensor[self._rows.i_prev.index, :]
        else:
            return self._tensor[self._rows.i_prev.index, self._model_mask]

    @i_prev.setter
    def i_prev(self, v):
        if self._model_mask is None:
            self._tensor[self._rows.i_prev.index, :] = v
        else:
            self._tensor[self._rows.i_prev.index, self._model_mask] = v




