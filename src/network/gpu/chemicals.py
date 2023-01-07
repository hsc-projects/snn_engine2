from typing import Optional, Union

# from network.network_config import NetworkConfig  # , PlottingConfig
from network.gpu.visualized_elements.chemical_concentration_volume_visual import ChemicalConcentrationVolumeVisual
from rendering import (
    GPUArrayCollection
)
# from network.chemical_config import ChemicalConfigCollection, DefaultChemicals


class ChemicalRepresentation(GPUArrayCollection):

    def __init__(self, network_shape: tuple,
                 scene, view,
                 device: int,
                 # plotting_config: PlottingConfig,
                 ):

        super().__init__(device=device, bprint_allocated_memory=False)

        if not hasattr(self, 'elements'):
            from network.chemical_config import ChemicalConfig
            self.elements: Optional[list[ChemicalConfig]] = None

        for el in self.elements:
            el.visual = ChemicalConcentrationVolumeVisual(None, network_shape,
                el.name, scene, view, device)

        print()
