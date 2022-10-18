from dataclasses import dataclass
from vispy import gloo
import sys

from engine import Engine

from pre_build_networks.io_network0 import IOSnn0Config
from pre_build_networks.rate_network0 import RateNetwork0Config


# high priority

# TODO: ReLu-ANN
# TODO: Synapse Visualization (incl. weight?)
# TODO: Chemical Diffusion + Visualization
# TODO: in-engine configuration/construction
# TODO: -ax**2 + b connection probability
# TODO: 1. minimalistic network; 2. rate network
# TODO: 1. concentration volumes; 2. concentration diffusions

# medium priority

# TODO: configurable segmentation
# TODO: subgroups

# low priority

# # simulation

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

# # ui

# TODO: gpu side group_info_mesh face color actualization
# TODO: group selection via selector box
# TODO: group selection via click


if __name__ == '__main__':

    gloo.gl.use_gl('gl+')
    eng = Engine(IOSnn0Config())
    # eng = Engine(RateNetwork0Config())
    if sys.flags.interactive != 1:
        eng.run()
