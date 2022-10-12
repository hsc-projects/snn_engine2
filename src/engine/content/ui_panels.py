from dataclasses import dataclass, asdict
from typing import Optional

from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QMenuBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from engine.content.widgets.combobox_frame import ComboBoxFrame, G2GInfoDualComboBoxFrame
from engine.content.widgets.gui_element import (
    ButtonMenuAction
)
from engine.content.widgets.rendered_object_collapsible import RenderedObjectCollapsible
from .neuron_properties_collapsible import SingleNeuronCollapsible, SingleNeuronCollapsibleContainer
from engine.content.widgets.collapsible_widget.collapsible_widget import CollapsibleWidget
from .widgets.spin_box_sliders import SpinBoxSlider
from network import SpikingNeuralNetwork, PlottingConfig
from network.network_state import MultiModelNeuronStateTensor


@dataclass
class ButtonMenuActions:

    """

    Declarative style. Must be initialized once.

    """

    window: Optional[QMainWindow] = None

    START_SIMULATION: ButtonMenuAction = ButtonMenuAction(menu_name='&Start Simulation',
                                                          menu_short_cut='F9',
                                                          status_tip='Start Simulation',
                                                          icon_name='control.png')

    PAUSE_SIMULATION: ButtonMenuAction = ButtonMenuAction(menu_name='&Pause Simulation',
                                                          menu_short_cut='F10',
                                                          status_tip='Pause Simulation',
                                                          icon_name='control-pause.png',
                                                          disabled=True)

    EXIT_APP: ButtonMenuAction = ButtonMenuAction(menu_name='&Exit',
                                                  name='Exit',
                                                  status_tip='Close Application',
                                                  menu_short_cut='Ctrl+Q')

    TOGGLE_OUTERGRID: ButtonMenuAction = ButtonMenuAction(menu_name='&OuterGrid',
                                                          name='Show OuterGrid',
                                                          status_tip='Show/Hide OuterGrid',
                                                          menu_short_cut='Ctrl+G',
                                                          checkable=True)

    ADD_SELECTOR_BOX: ButtonMenuAction = ButtonMenuAction(menu_name='&Add SelectorBox',
                                                          name='Add SelectorBox',
                                                          status_tip='Add SelectorBox')

    ACTUALIZE_G_FLAGS_TEXT: ButtonMenuAction = ButtonMenuAction(menu_name='&Refresh displayed G_flags',
                                                                menu_short_cut='F7',
                                                                icon_name='arrow-circle.png',
                                                                status_tip='Refresh displayed G_flags values')

    ACTUALIZE_G_PROPS_TEXT: ButtonMenuAction = ButtonMenuAction(menu_name='&Refresh displayed G2G_info',
                                                                menu_short_cut='F6',
                                                                icon_name='arrow-circle.png',
                                                                status_tip='Refresh displayed G2G_info values')

    ACTUALIZE_G2G_INFO_TEXT: ButtonMenuAction = ButtonMenuAction(
        menu_short_cut='F5',
        menu_name='&Refresh displayed G2G_flags ',
        icon_name='arrow-circle.png',
        status_tip='Refresh displayed G2G_flags values')

    TOGGLE_GROUP_IDS_TEXT: ButtonMenuAction = ButtonMenuAction(menu_name='&Group IDs',
                                                               menu_short_cut='Ctrl+F8',
                                                               checkable=True,
                                                               status_tip='Show/Hide Group IDs')

    TOGGLE_G_FLAGS_TEXT: ButtonMenuAction = ButtonMenuAction(menu_name='&G_flags Text',
                                                             checkable=True,
                                                             menu_short_cut='Ctrl+F7',
                                                             status_tip='Show/Hide G_flags values')

    TOGGLE_G_PROPS_TEXT: ButtonMenuAction = ButtonMenuAction(menu_name='&G_Props Text',
                                                             checkable=True,
                                                             menu_short_cut='Ctrl+F6',
                                                             status_tip='Show/Hide G_props values')

    TOGGLE_G2G_INFO_TEXT: ButtonMenuAction = ButtonMenuAction(menu_name='&G2G_info Text',
                                                              checkable=True,
                                                              menu_short_cut='Ctrl+F5',
                                                              status_tip='Show/Hide G2G_info values')

    def __post_init__(self):
        window_ = self.window
        self.window = None
        dct = asdict(self)
        self.window = window_
        print()
        for k in dct:
            v = getattr(self, k)
            if isinstance(v, ButtonMenuAction):
                if (v.menu_short_cut is not None) and (v.status_tip is not None):
                    v.status_tip = v.status_tip + f" ({v.menu_short_cut})"
                    print(v.status_tip)
                if v.window is None:
                    v.window = self.window


class MenuBar(QMenuBar):
    class MenuActions:
        def __init__(self):
            self.start: QAction = ButtonMenuActions.START_SIMULATION.action()
            self.pause: QAction = ButtonMenuActions.PAUSE_SIMULATION.action()
            self.toggle_outergrid: QAction = ButtonMenuActions.TOGGLE_OUTERGRID.action()
            self.exit: QAction = ButtonMenuActions.EXIT_APP.action()

            self.add_selector_box: QAction = ButtonMenuActions.ADD_SELECTOR_BOX.action()

            self.toggle_groups_ids: QAction = ButtonMenuActions.TOGGLE_GROUP_IDS_TEXT.action()
            self.toggle_g_flags: QAction = ButtonMenuActions.TOGGLE_G_FLAGS_TEXT.action()
            self.toggle_g_props: QAction = ButtonMenuActions.TOGGLE_G_PROPS_TEXT.action()
            self.toggle_g2g_info: QAction = ButtonMenuActions.TOGGLE_G2G_INFO_TEXT.action()

            self.actualize_g_flags: QAction = ButtonMenuActions.ACTUALIZE_G_FLAGS_TEXT.action()
            self.actualize_g_props: QAction = ButtonMenuActions.ACTUALIZE_G_PROPS_TEXT.action()
            self.actualize_g2g_info: QAction = ButtonMenuActions.ACTUALIZE_G2G_INFO_TEXT.action()

    def __init__(self, window):

        super().__init__(window)
        self.actions = self.MenuActions()

        self.setGeometry(QRect(0, 0, 440, 130))
        self.setObjectName("menubar")

        self.file_menu = self.addMenu('&File')
        self.file_menu.addAction(self.actions.start)
        self.file_menu.addAction(self.actions.pause)
        self.file_menu.addAction(self.actions.exit)

        self.objects_menu = self.addMenu('&Objects')
        self.objects_menu.addAction(self.actions.add_selector_box)

        self.view_menu = self.addMenu('&View')
        self.view_menu.addAction(self.actions.toggle_outergrid)

        self.view_menu.addAction(self.actions.toggle_groups_ids)

        self.view_menu.addAction(self.actions.toggle_g_flags)
        self.view_menu.addAction(self.actions.actualize_g_flags)
        self.view_menu.addAction(self.actions.toggle_g_props)
        self.view_menu.addAction(self.actions.actualize_g_props)
        self.view_menu.addAction(self.actions.toggle_g2g_info)
        self.view_menu.addAction(self.actions.actualize_g2g_info)


class UIPanel(QScrollArea):

    def __init__(self, window):
        super().__init__(window.centralWidget())
        self.setWidgetResizable(True)

        self.setWidget(QWidget(self))
        self.widget().setLayout(QVBoxLayout())

        self.widget().layout().setAlignment(Qt.AlignmentFlag.AlignTop)
        self.widget().layout().setSpacing(2)

    # noinspection PyPep8Naming
    def addWidget(self, *args):
        self.widget().layout().addWidget(*args)


class NeuronsCollapsible(CollapsibleWidget, SingleNeuronCollapsibleContainer):

    def __init__(self, title, parent=None):

        CollapsibleWidget.__init__(self, title=title, parent=parent)
        SingleNeuronCollapsibleContainer.__init__(self)

    def add_interfaced_neuron(self, network: SpikingNeuralNetwork, app,
                              title=None, neuron_model=MultiModelNeuronStateTensor,
                              neuron_id: Optional[int] = None, preset=None, window=None):
        neuron_collapsible = super().add_interfaced_neuron(
            title=title, window=window, network=network, app=app,
            neuron_model=neuron_model, neuron_id=neuron_id, preset=preset)

        self.add(neuron_collapsible)
        self.toggle_collapsed()
        self.toggle_collapsed()


class MainUILeft(UIPanel):

    class Buttons:
        def __init__(self):
            max_width = 140
            self.start: QPushButton = ButtonMenuActions.START_SIMULATION.button()
            self.pause: QPushButton = ButtonMenuActions.PAUSE_SIMULATION.button()
            self.exit: QPushButton = ButtonMenuActions.EXIT_APP.button()
            self.add_selector_box: QPushButton = ButtonMenuActions.ADD_SELECTOR_BOX.button()
            self.toggle_outergrid: QPushButton = ButtonMenuActions.TOGGLE_OUTERGRID.button()

            self.toggle_outergrid.setMinimumWidth(max_width)
            self.toggle_outergrid.setMaximumWidth(max_width)
            self.start.setMaximumWidth(max_width)
            self.exit.setMaximumWidth(max_width)

    class Sliders:
        def __init__(self, window):

            self.thalamic_inh_input_current = SpinBoxSlider(name='Inhibitory Current [I]',
                                                            window=window,
                                                            status_tip='Thalamic Inhibitory Input Current [I]',
                                                            prop_id='thalamic_inh_input_current',
                                                            maximum_width=300,
                                                            _min_value=0, _max_value=250)

            self.thalamic_exc_input_current = SpinBoxSlider(name='Excitatory Current [I]',
                                                            window=window,
                                                            status_tip='Thalamic Excitatory Input Current [I]',
                                                            prop_id='thalamic_exc_input_current',
                                                            maximum_width=300,
                                                            _min_value=0, _max_value=250)

            self.sensory_input_current0 = SpinBoxSlider(name='Input Current 0 [I]',
                                                        window=window,
                                                        status_tip='Sensory Input Current 0 [I]',
                                                        prop_id='sensory_input_current0',
                                                        maximum_width=300,
                                                        _min_value=0, _max_value=200)

            self.sensory_input_current1 = SpinBoxSlider(name='Input Current 1 [I]',
                                                        window=window,
                                                        status_tip='Sensory Input Current 1 [I]',
                                                        prop_id='sensory_input_current1',
                                                        maximum_width=300,
                                                        _min_value=0, _max_value=200)

            self.sensory_weight = SpinBoxSlider(name='Sensory',
                                                boxlayout_orientation=Qt.Orientation.Horizontal,
                                                window=window,
                                                func_=lambda x: float(x) / 100000 if x is not None else x,
                                                func_inv_=lambda x: int(x * 100000) if x is not None else x,
                                                status_tip='Sensory Weight',
                                                prop_id='src_weight',
                                                maximum_width=300,
                                                single_step_spin_box=0.01,
                                                single_step_slider=100,
                                                _min_value=0, _max_value=5)

    def __init__(self, window, plotting_config: PlottingConfig):

        UIPanel.__init__(self, window)

        self.window = window

        self.buttons = self.Buttons()
        self.sliders = self.Sliders(window)

        play_pause_widget = QWidget(self)
        play_pause_widget.setFixedSize(95, 45)
        play_pause_hbox = QHBoxLayout(play_pause_widget)
        play_pause_hbox.setContentsMargins(0, 0, 0, 0)
        play_pause_hbox.setSpacing(2)
        play_pause_hbox.addWidget(self.buttons.start)
        play_pause_hbox.addWidget(self.buttons.pause)

        # self.neuron0: Optional[SingleNeuronCollapsible] = None
        # self.neuron1 = None
        if plotting_config.windowed_neuron_interfaces is False:
            self.neurons_collapsible = NeuronsCollapsible(parent=self, title='Neuron Info')
        self.sensory_input_collapsible = CollapsibleWidget(self, title='Sensory Input')
        self.sensory_input_collapsible.add(self.sliders.sensory_input_current0.widget)
        self.sensory_input_collapsible.add(self.sliders.sensory_input_current1.widget)

        self.thalamic_input_collapsible = CollapsibleWidget(self, title='Thalamic Input')
        self.thalamic_input_collapsible.add(self.sliders.thalamic_inh_input_current.widget)
        self.thalamic_input_collapsible.add(self.sliders.thalamic_exc_input_current.widget)

        self.weights_collapsible = CollapsibleWidget(self, title='Weights')
        self.weights_collapsible.add(self.sliders.sensory_weight.widget)

        self.objects_collapsible = CollapsibleWidget(self, title='Objects')

        self.addWidget(play_pause_widget)
        self.addWidget(self.buttons.toggle_outergrid)
        if plotting_config.windowed_neuron_interfaces is False:
            self.addWidget(self.neurons_collapsible)
        self.addWidget(self.weights_collapsible)
        self.addWidget(self.sensory_input_collapsible)
        self.addWidget(self.thalamic_input_collapsible)

        self.addWidget(self.buttons.add_selector_box)
        self.addWidget(self.objects_collapsible)

        self.addWidget(self.buttons.exit)

        # self.interfaced_neurons: list[SingleNeuronCollapsible] = []
        # self.plotting_config: Optional[PlottingConfig] = plotting_config

    def add_3d_object_sliders(self, obj):
        collapsible = RenderedObjectCollapsible(obj, self.window, self)
        self.objects_collapsible.add(collapsible)
        return collapsible


class GroupInfoPanel(UIPanel):

    def __init__(self, window):

        super().__init__(window)

        self.combo_boxes_collapsible0 = CollapsibleWidget(title='Group Info Display 0')

        self.combo_boxes = []

        self.group_ids_combobox = ComboBoxFrame('Group IDs')
        self.add_combo_box(self.group_ids_combobox)

        self.g_flags_combobox = ComboBoxFrame(
            'G_flags', ButtonMenuActions.ACTUALIZE_G_FLAGS_TEXT.button())
        self.add_combo_box(self.g_flags_combobox)

        self.g_props_combobox = ComboBoxFrame(
            'G_props', ButtonMenuActions.ACTUALIZE_G_PROPS_TEXT.button())
        self.add_combo_box(self.g_props_combobox)

        self.combo_boxes_collapsible1 = CollapsibleWidget(title='Group Info Display 1')
        self.g2g_info_combo_box = G2GInfoDualComboBoxFrame(
            'G2G_info', ButtonMenuActions.ACTUALIZE_G2G_INFO_TEXT.button())
        self.combo_boxes_collapsible1.add(self.g2g_info_combo_box)

        self.addWidget(self.combo_boxes_collapsible0)
        self.addWidget(self.combo_boxes_collapsible1)

    def add_combo_box(self, combo_box):

        self.combo_boxes_collapsible0.add(combo_box)
        # self.addWidget(combo_box)
        self.combo_boxes.append(combo_box)


# class NeuronsFrame(QFrame):
#
#     def __init__(self, parent):
#
#         QFrame.__init__(self, parent)
#

class NeuronInterfacePanel(UIPanel, SingleNeuronCollapsibleContainer):

    def __init__(self, window, n_columns=2):

        UIPanel.__init__(self, window=window)
        SingleNeuronCollapsibleContainer.__init__(self, )

        self.interfaced_neurons: list[SingleNeuronCollapsible] = []
        self.single_neuron_plot_length: Optional[int] = None

        self.n_frame_cols = n_columns
        self.last_frame_col = n_columns - 1
        self.frame_cols = []

        self.setLayout(QHBoxLayout())

        for i in range(self.n_frame_cols):
            frame = UIPanel(window=window)
            # frame.setLayout(QVBoxLayout())
            self.layout().addWidget(frame)
            self.frame_cols.append(frame)

    def add_interfaced_neuron(self, network: SpikingNeuralNetwork, app,
                              title=None, neuron_model=MultiModelNeuronStateTensor,
                              neuron_id: Optional[int] = None, preset=None, window=None,
                              frame_column=None):
        neuron_collapsible = super().add_interfaced_neuron(
            title=title, window=window, network=network, app=app,
            neuron_model=neuron_model, neuron_id=neuron_id, preset=preset)
        if frame_column is None:
            self.last_frame_col = (self.last_frame_col + 1) % self.n_frame_cols
            frame_column = self.last_frame_col

        self.frame_cols[frame_column].addWidget(neuron_collapsible)
        neuron_collapsible.toggle_collapsed()
