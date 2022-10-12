from dataclasses import dataclass
import numba.cuda
from vispy import gloo
import sys

from network import (
    NetworkInitValues,
    NetworkConfig,
    PlottingConfig,
    SpikingNeuronNetwork
)
from app import BaseApp

# TODO: pre-synaptic delays
# TODO: resonant cells/groups,
# TODO: better sensory weights,
# TODO: filter-cells,
# TODO: group_info_mesh face sizes
# TODO: better stdp G2G config
# TODO: monitor learning
# TODO: weird synapse counts
# TODO: low neuron count swaps
# TODO: performance hit above 300K neurons

# TODO (optional) gpu side group_info_mesh face color actualization
# TODO (optional) group selection via selector box
# TODO (optional, difficult) group selection via click


# TODO: configurable segmentation
# tODO: subgroups

class EngineConfig:

    class InitValues(NetworkInitValues):

        @dataclass
        class ThalamicInput:
            inh_current: float = 25.
            exc_current: float = 15.

        @dataclass
        class SensoryInput:
            input_current0: float = 22.3
            input_current1: float = 42.8

        @dataclass
        class Weights:
            Inh2Exc: float = -.49
            Exc2Inh: float = .75
            Exc2Exc: float = .75
            SensorySource: float = 1.5

    N: int = 25 * 10 ** 3
    T: int = 5000  # Max simulation record duration

    device: int = 0

    max_batch_size_mb: int = 300

    network = NetworkConfig(N=N,
                            N_pos_shape=(4, 4, 1),
                            sim_updates_per_frame=1,
                            stdp_active=True,
                            debug=False, InitValues=InitValues())
    plotting = PlottingConfig(n_voltage_plots=10, voltage_plot_length=200,
                              n_scatter_plots=10, scatter_plot_length=200,
                              windowed_multi_neuron_plots=False,
                              windowed_neuron_interfaces=False,
                              group_info_view_mode='split',
                              network_config=network)


class Engine(BaseApp):

    def __init__(self, config=None):

        # noinspection PyUnresolvedReferences
        from pycuda import autoinit

        self.config = EngineConfig() if config is None else config

        numba.cuda.select_device(self.config.device)

        # keep order for vbo id (1/4)
        super().__init__(self.config.plotting)
        # keep order for vbo id (2/4)
        self.network = SpikingNeuronNetwork(self.config)
        # keep order for vbo id (3/4)
        self.network.initialize_GPU_arrays(self.config.device, self)
        # keep order (4/4)
        self._bind_ui()


if __name__ == '__main__':

    gloo.gl.use_gl('gl+')
    eng = Engine()
    if sys.flags.interactive != 1:
        eng.run()
