from dataclasses import dataclass, asdict, field
from typing import Optional, List
from network.gpu.chemicals import ChemicalConcentrations
from network.gpu.visualized_elements.chemical_concentration_volume_visual import ChemicalConcentrationVolumeVisual


@dataclass
class ChemicalConfig:
    name: Optional[str] = None
    color: Optional[str] = None
    effect: Optional[str] = None
    visual: Optional[ChemicalConcentrationVolumeVisual] = None
    k_val: float = 0.16
    depreciation: float = 0.0


@dataclass
class ChemicalConfigCollection(ChemicalConcentrations):

    _elements: Optional[List] = field(init=False, repr=False)

    def __post_init__(self):
        self._elements = []
        dict_ = asdict(self)
        for k in dict_.keys():
            if k != '_elements':
                el = getattr(self, k)
                if el.name is None:
                    setattr(el, 'name', k)
                self._elements.append(el)

    def __iter__(self):
        for el in self._elements:
            yield el

    def __setattr__(self, key, value):
        if hasattr(self, '_elements'):
            if key in self.names:
                raise AttributeError("Resetting of 'ChemicalConfig' attributes is prohibited.")
            if isinstance(value, ChemicalConfig):
                raise AttributeError("Setting of 'ChemicalConfig' attributes is prohibited.")
        super().__setattr__(key, value)

    @property
    def elements(self) -> list[ChemicalConfig]:
        return self._elements

    @property
    def names(self) -> list[str]:
        return [x.name for x in self._elements]

    def super_init(self, network_shape, scene, view, device):
        super().__init__(network_shape, scene, view, device)
        return self

    # def validate(self):
    #     for i, k in enumerate(self.names):
    #         assert isinstance(getattr(self, k), ChemicalConfig)
    #         assert


@dataclass
class DefaultChemicals(ChemicalConfigCollection):

    C1: ChemicalConfig = ChemicalConfig()
    # C2: ChemicalConfig = ChemicalConfig()
