from dataclasses import dataclass

from network import (
    NetworkConfig,
    PlottingConfig,
    SpikingNeuralNetwork,
)
from network.gpu.visualized_elements import (
    GroupFiringCountsPlot,
)
from network.gpu.neurons import NeuronRepresentation
from network.gpu.synapses import SynapseRepresentation
from engine import EngineConfig, Engine
from network.gpu.simulation import NetworkSimulationGPU
from network.network_config import NetworkInitValues


class RateNetwork0GPU(NetworkSimulationGPU):

    def __init__(self,
                 engine,
                 config: NetworkConfig,
                 neurons: NeuronRepresentation,
                 synapses: SynapseRepresentation,
                 device: int,
                 T: int,
                 plotting_config: PlottingConfig,
                 group_firing_counts_plot_single1: GroupFiringCountsPlot
                 ):

        super().__init__(engine=engine, config=config,
                         neurons=neurons,
                         synapses=synapses,
                         device=device, T=T,
                         plotting_config=plotting_config,
                         group_firing_counts_plot_single1=group_firing_counts_plot_single1)

        self._post_synapse_mod_init(self.neurons.data_shapes)


class RateNetwork0(SpikingNeuralNetwork):

    def __init__(self, config, engine):
        super().__init__(config, engine)

    # noinspection PyPep8Naming
    def initialize_GPU_arrays(self, device, engine: Engine, **kwargs):
        super().initialize_GPU_arrays(device, engine)

        self.simulation_gpu = RateNetwork0GPU(
            engine=engine,
            config=self.network_config,
            neurons=self.neurons,
            synapses=self.synapse_arrays,
            device=device,
            T=self.T,
            plotting_config=self.plotting_config,
            group_firing_counts_plot_single1=self.group_firing_counts_plot_single1
        )

        self.registered_buffers += self.simulation_gpu.registered_buffers

        self.synapse_arrays.visualized_synapses.add_synapse_visual(0)

    def unregister_registered_buffers(self):
        super().unregister_registered_buffers()


class RateNetwork0Config(EngineConfig):

    class InitValues(NetworkInitValues):

        @dataclass
        class Weights:
            Inh2Exc: float = '-r'
            Exc2Inh: float = 'r'
            Exc2Exc: float = 'r'
            SensorySource: float = 1.5

    N: int = 5 * 10 ** 2
    T: int = 5000  # Max simulation record duration

    device: int = 0

    max_batch_size_mb: int = 300

    network = NetworkConfig(N=N,
                            N_pos_shape=(4, 4, 1),
                            sim_updates_per_frame=1,
                            stdp_active=True,
                            debug=False, InitValues=InitValues())

    plotting = PlottingConfig(n_voltage_plots=10,
                              voltage_plot_length=200,
                              n_scatter_plots=10,
                              scatter_plot_length=200,
                              has_voltage_multiplot=True,
                              has_firing_scatterplot=True,
                              has_group_firings_multiplot=True,
                              has_group_firings_plot0=True,
                              has_group_firings_plot1=True,
                              windowed_multi_neuron_plots=False,
                              windowed_neuron_interfaces=True,
                              group_info_view_mode='split',
                              network_config=network)

    network_class = RateNetwork0
