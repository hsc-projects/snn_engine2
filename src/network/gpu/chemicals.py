import torch
from typing import Optional

# from network.network_config import NetworkConfig  # , PlottingConfig
from network.gpu.visualized_elements.chemical_concentration_volume_visual import ChemicalConcentrationVolumeVisual
from rendering import (
    GPUArrayCollection,
    RegisteredTexture3D
)
# from network.chemical_config import ChemicalConfigCollection, DefaultChemicals


class ChemicalConcentrations(GPUArrayCollection):

    def __init__(self, network_shape: tuple,
                 scene, view,
                 device: int,
                 # plotting_config: PlottingConfig,
                 ):

        super().__init__(device=device, bprint_allocated_memory=False)

        if not hasattr(self, 'elements'):
            from network.chemical_config import ChemicalConfig
            self.elements: Optional[list[ChemicalConfig]] = None

        self.textures3D: list[RegisteredTexture3D] = []
        self._has_elements = False

        for el in self.elements:
            el.visual = ChemicalConcentrationVolumeVisual(None, network_shape,
                el.name, scene, view, device)
            self.textures3D.append(el.visual.gpu_array)

        if len(self.elements) > 0:
            self._has_elements = True
            self.C_new = self.textures3D[0].tensor
            # self.C_new[:] = 200
            # self.C_new[:, :, :-2] = 5
            self.textures3D[0].cpy_tnsr2tex()
        else:
            self.C_new = self.fzeros((0, 0, 0))

        self.C_old = torch.clone(self.C_new)
        self.C_source = self.fzeros(shape=self.C_new.shape)
        if self._has_elements is True:
            self.C_source[:, :, -1] = 2000
            mask = self.C_new >= self.C_new.max()-200
            self.C_source[mask] = self.C_new[mask]

            # print(self.C_new[100, 0, 60:70])
            # print(self.C_new.min(), self.C_new.max())
            # print(self.C_new.shape)

    def update_visuals(self):
        # print(self.C_new[100, 0, 60:70])
        # print(self.C_new.mean())
        self.textures3D[0].cpy_tnsr2tex()

    @property
    def depth(self):
        return self.C_new.shape[1]

    @property
    def height(self):
        return self.C_new.shape[0]

    @property
    def k_val(self):
        if self._has_elements is False:
            return 0.0
        return self.elements[0].k_val

    @property
    def depreciation(self):
        if self._has_elements is False:
            return 0.0
        return self.elements[0].depreciation

    @property
    def width(self):
        return self.C_new.shape[2]
