from PyQt6.QtWidgets import (
    QLabel,
    QPushButton,
    QSpinBox
)
from typing import Union

from .widgets.spin_box_sliders import SubCollapsibleFrame
from .widgets.gui_element import PushButton
from engine.content.widgets.collapsible_widget.collapsible_widget import CollapsibleWidget


from network.chemical_config import DefaultChemicals, ChemicalConfigCollection, ChemicalConfig


class ButtonFrame(SubCollapsibleFrame):

    def __init__(self, parent, el: ChemicalConfig, fixed_width=300, label='ID'):

        super().__init__(parent, fixed_width=fixed_width)

        self.button_visible: QPushButton = PushButton('visible', self)
        self.button_visible.setCheckable(True)
        self.button_visible.setChecked(True)
        self.button_visible.clicked.connect(el.visual.toggle_visible)

        self.button_add_10: QPushButton = PushButton('add', self)
        self.button_add_10.clicked.connect(el.visual.add)

        self.spinbox = QSpinBox(self)
        self.spinbox.setMinimum(-1000)
        self.spinbox.setMaximum(1000)

        self.layout().addWidget(self.button_visible)
        self.layout().addWidget(self.spinbox)
        self.layout().addWidget(self.button_add_10)

        self.setFixedHeight(28)


class ChemicalControlCollapsible(CollapsibleWidget):

    def __init__(self, parent, el: ChemicalConfig,):

        super().__init__(parent=parent, title=el.name)

        self.connected_chemical: ChemicalConfig = el
        self.button_frame = ButtonFrame(self, el)

        self.button_frame.spinbox.setValue(self.connected_chemical.visual.additive)
        self.button_frame.spinbox.valueChanged.connect(self.update_additive)

        self.add(self.button_frame)

    def update_additive(self):
        self.connected_chemical.visual.additive = self.button_frame.spinbox.value()


class ChemicalControlCollapsibleContainer(CollapsibleWidget):

    def __init__(self,  # chemical_collection: Union[DefaultChemicals, ChemicalConfigCollection],
                 title='Chemical Control', parent=None):

        CollapsibleWidget.__init__(self, title=title, parent=parent)

    def add_chemical_control_collapsibles(self, chemical_collection: Union[DefaultChemicals, ChemicalConfigCollection]):
        for el in chemical_collection:
            self.add_chemical_control_collapsible(el)

    def add_chemical_control_collapsible(self, el: ChemicalConfig):
        chem_collapsible = ChemicalControlCollapsible(self, el=el)
        self.add(chem_collapsible)
        self.toggle_collapsed()
        self.toggle_collapsed()
