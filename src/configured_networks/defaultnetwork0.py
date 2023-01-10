from dataclasses import dataclass

from network import (
    NetworkConfig,
    PlottingConfig,
    SpikingNeuralNetwork,
)
from engine import EngineConfig, Engine
from network.network_config import NetworkInitValues, DefaultChemicals
from utils import boxed_string


class DefaultNetwork0(SpikingNeuralNetwork):

    def __init__(self, config, engine):
        super().__init__(config, engine)

    # noinspection PyPep8Naming
    def initialize_GPU_arrays(self, device, engine: Engine, init_default_sim=True,
                              init_default_sim_with_syn_post_init=True, **kwargs):
        super().initialize_GPU_arrays(device, engine,
                                      init_default_sim=init_default_sim,
                                      init_default_sim_with_syn_post_init=init_default_sim_with_syn_post_init)

    def unregister_registered_buffers(self):
        super().unregister_registered_buffers()


class DefaultNetwork0Config(EngineConfig):

    network_class = DefaultNetwork0
    # print('\n', boxed_string(DefaultNetwork0.__name__, inbox_h_margin=20))

    class InitValues(NetworkInitValues):

        @dataclass
        class Weights:
            Inh2Exc: float = '-r'  # 'r' := random
            Exc2Inh: float = 'r'
            Exc2Exc: float = 'r'
            SensorySource: float = 1.5

    N: int = 5 * 10 ** 3
    T: int = 5000  # Max simulation record duration

    device: int = 0

    max_batch_size_mb: int = 300

    network = NetworkConfig(N=N,
                            N_pos_shape=(4, 4, 1),
                            sim_updates_per_frame=200,
                            stdp_active=True,
                            debug=False, InitValues=InitValues(),
                            chemical_configs=DefaultChemicals())

    plotting = PlottingConfig(n_voltage_plots=10,
                              voltage_plot_length=200,
                              n_scatter_plots=10,
                              scatter_plot_length=200,
                              has_voltage_multiplot=True,
                              has_firing_scatterplot=True,
                              has_group_firings_multiplot=False,
                              has_group_firings_plot0=False,
                              has_group_firings_plot1=False,
                              windowed_multi_neuron_plots=True,
                              windowed_neuron_interfaces=False,
                              group_info_view_mode='windowed',
                              network_config=network)

