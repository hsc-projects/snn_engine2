from dataclasses import dataclass, field, InitVar
from typing import Optional, Union

import numpy as np
from scipy import signal


class SignalModel:

    class VariableConfig:
        pass


@dataclass
class BaseSignalVariable:
    interval: list
    step_size: Union[int, float]
    unit: Optional[str] = None


@dataclass
class AmplitudeVariable(BaseSignalVariable):
    interval: list = field(default_factory=lambda: [0, 500])
    step_size: float = 0.1
    unit: str = 'pA'


@dataclass
class PeriodVariable(BaseSignalVariable):
    interval: list = field(default_factory=lambda: [1, 1000])
    step_size: int = 1
    unit: str = 'ms'


@dataclass
class PhaseVariable(BaseSignalVariable):
    interval: list = field(default_factory=lambda: [-1000, 1000])
    step_size: int = 1
    unit: str = 'ms'


@dataclass
class StepTimeVariable(BaseSignalVariable):
    interval: list = field(default_factory=lambda: [0, 100000])
    step_size: int = 1
    unit: str = 'ms'


@dataclass
class OffsetVariable(BaseSignalVariable):
    interval: list = field(default_factory=lambda: [-500, 500])
    step_size: float = 0.1
    unit: str = 'pA'


@dataclass
class FrequencyVariable(BaseSignalVariable):
    interval: list = field(default_factory=lambda: [0, 1000])
    step_size: float = 1
    unit: str = 'Hz'


class StepSignal(SignalModel):

    class VariableConfig:
        amplitude = AmplitudeVariable()
        step_time = StepTimeVariable()

    def __init__(self, amplitude, phase):

        self.amplitude = amplitude
        self.step_time = phase

    def __call__(self, t, t_mod):
        if t < self.step_time:
            return 0
        else:
            return self.amplitude


class PulseSignal(SignalModel):

    class VariableConfig:
        amplitude = AmplitudeVariable()
        period = PeriodVariable()
        duty = PeriodVariable()
        phase = PhaseVariable()

    def __init__(self, amplitude, phase, frequency, pulse_length, offset):
        self.amplitude = amplitude
        self.period = int(1000 / frequency)
        self.duty = round(pulse_length/self.period, 2)
        self.phase = phase
        self.offset = offset

    def __call__(self, t, t_mod):
        return (signal.square(((t + self.phase) * 2 * np.pi)/self.period, self.duty) * self.amplitude / 2
                + self.amplitude / 2) + self.offset


class DiscretePulseSignal(SignalModel):

    class VariableConfig:
        amplitude = AmplitudeVariable()
        period = PeriodVariable()
        duty_period = PeriodVariable()
        phase = PhaseVariable()
        offset = OffsetVariable()

    def __init__(self, amplitude, phase, frequency, pulse_length, offset=0):
        self.amplitude = amplitude
        self.period = int(1000 / frequency)
        self.duty_period = pulse_length
        self.offset = offset
        self.phase = phase

    def __call__(self, t, t_mod):
        return (((t + self.phase) % self.period) < self.duty_period) * self.amplitude + self.offset


class CosineSignal(SignalModel):

    class VariableConfig:
        amplitude = AmplitudeVariable()
        frequency = FrequencyVariable()
        phase = PhaseVariable()
        offset = OffsetVariable()

    def __init__(self, amplitude, phase, frequency, offset=0):
        self.amplitude = amplitude
        self.frequency = frequency
        self.offset = offset
        self.phase = phase

    def __call__(self, t, t_mod):
        return self.amplitude * np.cos(((2 * self.frequency * np.pi * t)/1000) + self.phase) + self.offset


class PeriodicSpikeTrain(SignalModel):

    class VariableConfig:
        amplitude = AmplitudeVariable()
        period = PeriodVariable()
        spike_period = PeriodVariable()
        duty_period = PeriodVariable()
        phase = PhaseVariable()
        offset = OffsetVariable()

    def __init__(self, amplitude, phase, positive_period_length, spike_period, period, frequency=None, offset=0):

        if frequency is not None:
            assert period is None
            self.period = int(1000 / frequency)
        else:
            assert period is not None
            self.period = period

        self.amplitude = amplitude

        self.spike_period = spike_period
        self.duty_period = positive_period_length
        self.offset = offset
        self.phase = phase

    def __call__(self, t, t_mod):

        if ((t + self.phase) % self.period) < self.duty_period:
            sign = 1
        else:
            sign = -1

        return sign * (((t + self.phase) % self.spike_period) == 0) * self.amplitude + self.offset


@dataclass
class SignalCollection:

    amplitude: InitVar[int] = 25
    frequency: InitVar[int] = 50
    period: InitVar[int] = 24
    pulse_length: InitVar[int] = 12
    spike_period_length: InitVar[int] = 3
    phase: InitVar[int] = 0
    offset: InitVar[int] = 0

    step_signal: Optional[StepSignal] = None
    discrete_pulse: Optional[DiscretePulseSignal] = None
    cosine_signal: Optional[CosineSignal] = None
    periodic_spike_train: Optional[PeriodicSpikeTrain] = None

    def __post_init__(self, amplitude: int,
                      frequency: int,
                      period: int,
                      pulse_length: int,
                      spike_period_length: int,
                      phase: int,
                      offset: int):

        if self.step_signal is None:
            self.step_signal = StepSignal(amplitude=amplitude, phase=phase)
        if self.discrete_pulse is None:
            self.discrete_pulse = DiscretePulseSignal(
                amplitude=amplitude, phase=phase, frequency=frequency,
                pulse_length=pulse_length, offset=offset)
        if self.cosine_signal is None:
            self.cosine_signal = CosineSignal(
                amplitude=amplitude, frequency=frequency, phase=phase, offset=offset)
        if self.periodic_spike_train is None:
            self.periodic_spike_train = PeriodicSpikeTrain(
                amplitude=amplitude, phase=phase, period=period,
                positive_period_length=int(period/2),
                spike_period=spike_period_length, offset=offset)

