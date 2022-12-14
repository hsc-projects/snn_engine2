from dataclasses import dataclass, asdict
import numpy as np
from typing import Optional, Union

import pandas as pd

from .network_grid import NetworkGrid
from utils import boxed_string
from .chemical_config import ChemicalConfig, ChemicalConfigCollection, DefaultChemicals


class NetworkInitValues:
    @dataclass
    class ThalamicInput:
        inh_current: float = 25.
        exc_current: float = 15.

    @dataclass
    class SensoryInput:
        input_current0: float = 65.
        input_current1: float = 25.

    @dataclass
    class Weights:
        Inh2Exc: float = -.49
        Exc2Inh: float = .75
        Exc2Exc: float = .75
        SensorySource: float = .75

    def __init__(self):
        self.ThalamicInput = self.ThalamicInput()
        self.SensoryInput = self.SensoryInput()
        self.Weights = self.Weights()


@dataclass
class NetworkConfig:

    N: int = 100000
    S: Optional[int] = None
    D: Optional[int] = None
    G: Optional[int] = None

    pos: np.array = None
    N_pos_shape: tuple = (1, 1, 1)

    vispy_scatter_plot_stride: int = 13  # enforced by the vispy scatterplot memory layout

    grid_segmentation: tuple = None

    N_G_n_cols: int = 2
    N_G_neuron_type_col: int = 0
    N_G_group_id_col: int = 1

    sensory_groups: Optional[list[int]] = None
    output_groups: Optional[list[int]] = None

    max_z: float = 999.

    sim_updates_per_frame: Optional[int] = None

    debug: bool = False

    stdp_active: bool = False

    max_n_winner_take_all_layers: int = 20
    max_winner_take_all_layer_size: int = 10

    InitValues: NetworkInitValues = NetworkInitValues()

    grid: Optional[NetworkGrid] = None

    chemical_configs: Optional[Union[list[ChemicalConfig],
                                     ChemicalConfig,
                                     ChemicalConfigCollection,
                                     DefaultChemicals]] = None

    def __str__(self):
        name = self.__class__.__name__
        msg = boxed_string(name)
        msg += f'\n\tN={self.N}, \n\tS={self.S}, \n\tD={self.D}, \n\tG={self.G},'
        msg += f'\n\tN_pos_shape={self.N_pos_shape},'
        msg += f'\n\tgrid_segmentation={self.grid_segmentation}\n'
        return msg

    def __post_init__(self):

        assert self.N % 2 == 0

        if self.N <= 4000:
            self.N_pos_shape = (1, 1, 1)

        if self.sim_updates_per_frame is None:
            if self.N <= 10 ** 5:
                self.sim_updates_per_frame = 1
            elif self.N <= 40 ** 5:
                self.sim_updates_per_frame = 5
            else:
                self.sim_updates_per_frame = 10

        if self.S is None:
            self.S = int(min(1000, max(np.sqrt(self.N) + 50, 2)))
        if self.D is None:
            self.D = min(int(max(np.log10(self.N) * (1 + np.sqrt(np.log10(self.N))), 2)), 20)

        # assert self.N >= 20
        # assert self.N <= 2 * 10 ** 6
        assert isinstance(self.N, int)
        assert self.S <= 1000
        assert isinstance(self.S, int)
        assert self.D <= 100
        assert isinstance(self.D, int)
        assert len(self.N_pos_shape) == 3
        # assert self.is_cube
        # assert self.shape[0] == 1
        min_shape_el = min(self.N_pos_shape)
        assert all([
            isinstance(s, int) and (s / min_shape_el == int(s / min_shape_el)) for s in self.N_pos_shape
        ])
        assert self.vispy_scatter_plot_stride == 13  # enforced by the vispy scatterplot memory layout

        self.swap_tensor_shape_multiplicators: tuple = (self.S, 10)

        if self.grid is None:
            self.grid = NetworkGrid(self)

        if self.chemical_configs is not None:
            if not isinstance(self.chemical_configs, ChemicalConfigCollection):
                chemical_configs = self.chemical_configs
                if not isinstance(self.chemical_configs, list):
                    chemical_configs = [self.chemical_configs]
                names = []
                for chem in chemical_configs:
                    assert isinstance(chem, ChemicalConfig)
                    names.append(chem.name)
                assert not bool(pd.Series(names).duplicated().any())
                self.chemical_configs = ChemicalConfigCollection()
                for chem in chemical_configs:
                    setattr(self.chemical_configs, chem.name, chem)


@dataclass(frozen=True)
class PlotViewModeOption:

    mode: str

    def __post_init__(self):
        view_modes = ['scene', 'windowed', 'split']
        assert self.mode in view_modes

    def __eq__(self, other):
        return self.mode == other

    @property
    def scene(self):
        return self.mode == 'scene'

    @property
    def split(self):
        return self.mode == 'split'

    @property
    def windowed(self):
        return self.mode == 'windowed'


@dataclass
class PlottingConfig:

    n_voltage_plots: int
    voltage_plot_length: int

    n_scatter_plots: int
    scatter_plot_length: int

    network_config: NetworkConfig

    has_voltage_multiplot: bool = True
    has_firing_scatterplot: bool = True
    has_group_firings_multiplot: bool = True
    has_group_firings_plot0: bool = True
    has_group_firings_plot1: bool = True

    windowed_multi_neuron_plots: bool = True
    windowed_neuron_interfaces: bool = True
    group_info_view_mode: Optional[Union[PlotViewModeOption, str]] = 'split'

    _max_length: int = 10000
    _max_n_voltage_plots: int = 1000
    _max_n_scatter_plots: int = 1000

    def __post_init__(self):

        if self.windowed_multi_neuron_plots is True:
            assert self.has_voltage_multiplot
            assert self.has_firing_scatterplot

        if isinstance(self.group_info_view_mode, str):
            self.group_info_view_mode = PlotViewModeOption(self.group_info_view_mode)
        elif not isinstance(self.group_info_view_mode, PlotViewModeOption):
            raise TypeError

        self.n_voltage_plots = min(min(self.N, self.n_voltage_plots), self._max_n_voltage_plots)
        self.n_scatter_plots = min(min(self.N, self.n_scatter_plots), self._max_n_scatter_plots)
        self.voltage_plot_length = min(self.voltage_plot_length, self._max_length)
        self.scatter_plot_length = min(self.scatter_plot_length, self._max_length)

    # noinspection PyPep8Naming
    @property
    def N(self):
        return self.network_config.N

    # noinspection PyPep8Naming
    @property
    def G(self):
        return self.network_config.G


@dataclass
class BufferCollection:

    def __post_init__(self):
        for k, v in asdict(self).items():
            if v is not None:
                setattr(self, k, int(v))
