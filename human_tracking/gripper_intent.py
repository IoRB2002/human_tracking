from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import math
from human_tracking.tracking import HandDerivedKinematics, HumanDerivedKinematics

class HandSide(str, Enum):
    LEFT = "left"
    RIGHT = "right"

class GripperIntentState(str, Enum):
    UNKNOWN = "unknown"
    OPEN = "open"
    CLOSED = "closed"

@dataclass(frozen=True)
class GripperIntentConfig:
    close_threshold: float
    open_threshold: float
    consecutive_confirmations: int

    def __post_init__(self) -> None:
        for name, value in (
            ("close_threshold", self.close_threshold),
            ("open_threshold", self.open_threshold),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative.")

        if not self.close_threshold < self.open_threshold:
            raise ValueError("close_threshold must be lower than open_threshold to provide hysteresis.")

        if self.consecutive_confirmations < 1:
            raise ValueError("consecutive_confirmations must be at least 1.")

@dataclass(frozen=True)
class GripperIntentResult:
    frame_sequence_id: int
    measurement_timestamp_s: float
    hand_side: HandSide
    measurement_valid: bool
    valid_for_control: bool
    stable_state: GripperIntentState
    state_changed: bool
    desired_state: GripperIntentState | None
    pending_state: GripperIntentState | None
    consecutive_pending: int
    pinch_ratio: float | None
    reasons: tuple[str, ...]

class GripperIntentTracker:
    def __init__(self, hand_side: HandSide, config: GripperIntentConfig):
        self.hand_side = hand_side
        self.config = config
        self._stable_state = GripperIntentState.UNKNOWN
        self._pending_state: GripperIntentState | None = None
        self._consecutive_pending = 0
        self._last_frame_sequence_id: int | None = None
        self._last_timestamp_s: float | None = None

    @property
    def stable_state(self) -> GripperIntentState:
        return self._stable_state

    def reset(self) -> None:
        self._stable_state = GripperIntentState.UNKNOWN
        self._pending_state = None
        self._consecutive_pending = 0
        self._last_frame_sequence_id = None
        self._last_timestamp_s = None

    def update(self, human: HumanDerivedKinematics) -> GripperIntentResult:
        _validate_human_order(
            human,
            self._last_frame_sequence_id,
            self._last_timestamp_s,
            stream_name="Gripper-intent",
        )
        self._last_frame_sequence_id = human.frame_sequence_id
        self._last_timestamp_s = human.measurement_timestamp_s
        pinch_ratio, measurement_reasons = _pinch_ratio_measurement(human, self.hand_side)

        if measurement_reasons:
            self._clear_pending()
            return self._result(
                human,
                measurement_valid = False,
                state_changed = False,
                desired_state = None,
                pinch_ratio = pinch_ratio,
                reasons = measurement_reasons,
            )

        assert pinch_ratio is not None

        requested_state = self._requested_state(pinch_ratio)
        state_changed = False

        if requested_state is None or requested_state == self._stable_state:
            self._clear_pending()
        else:
            if self._pending_state == requested_state:
                self._consecutive_pending += 1
            else:
                self._pending_state = requested_state
                self._consecutive_pending = 1

            if self._consecutive_pending >= self.config.consecutive_confirmations:
                self._stable_state = requested_state
                state_changed = True
                self._clear_pending()

        desired_state = (
            None if self._stable_state == GripperIntentState.UNKNOWN
            else self._stable_state
        )

        return self._result(
            human,
            measurement_valid = True,
            state_changed = state_changed,
            desired_state = desired_state,
            pinch_ratio = pinch_ratio,
            reasons = (),
        )

    def _requested_state(self, pinch_ratio: float) -> GripperIntentState | None:
        if pinch_ratio <= self.config.close_threshold:
            return GripperIntentState.CLOSED
        if pinch_ratio >= self.config.open_threshold:
            return GripperIntentState.OPEN
        return None

    def _clear_pending(self) -> None:
        self._pending_state = None
        self._consecutive_pending = 0

    def _result(
        self,
        human: HumanDerivedKinematics,
        *,
        measurement_valid: bool,
        state_changed: bool,
        desired_state: GripperIntentState | None,
        pinch_ratio: float | None,
        reasons: tuple[str, ...],
    ) -> GripperIntentResult:
        return GripperIntentResult(
            frame_sequence_id = human.frame_sequence_id,
            measurement_timestamp_s = human.measurement_timestamp_s,
            hand_side = self.hand_side,
            measurement_valid = measurement_valid,
            valid_for_control = (measurement_valid and desired_state is not None),
            stable_state = self._stable_state,
            state_changed = state_changed,
            desired_state = desired_state,
            pending_state = self._pending_state,
            consecutive_pending = self._consecutive_pending,
            pinch_ratio = pinch_ratio,
            reasons = reasons,
        )

@dataclass(frozen=True)
class GripperApertureConfig:
    closed_reference_ratio: float
    open_reference_ratio: float

    def __post_init__(self) -> None:
        for name, value in (
            ("closed_reference_ratio", self.closed_reference_ratio),
            ("open_reference_ratio", self.open_reference_ratio),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative.")

        if not self.closed_reference_ratio < self.open_reference_ratio:
            raise ValueError("closed_reference_ratio must be lower than open_reference_ratio.")

@dataclass(frozen=True)
class GripperApertureResult:
    frame_sequence_id: int
    measurement_timestamp_s: float
    hand_side: HandSide
    measurement_valid: bool
    normalized_aperture: float | None
    fully_open: bool
    fully_closed: bool
    pinch_ratio: float | None
    reasons: tuple[str, ...]

class GripperApertureTracker:
    def __init__(self, hand_side: HandSide, config: GripperApertureConfig):
        self.hand_side = hand_side
        self.config = config
        self._last_frame_sequence_id: int | None = None
        self._last_timestamp_s: float | None = None

    def reset(self) -> None:
        self._last_frame_sequence_id = None
        self._last_timestamp_s = None

    def update(self, human: HumanDerivedKinematics) -> GripperApertureResult:
        _validate_human_order(
            human,
            self._last_frame_sequence_id,
            self._last_timestamp_s,
            stream_name="Gripper-aperture",
        )
        self._last_frame_sequence_id = human.frame_sequence_id
        self._last_timestamp_s = human.measurement_timestamp_s
        pinch_ratio, measurement_reasons = _pinch_ratio_measurement(human, self.hand_side)

        if measurement_reasons:
            return self._result(
                human,
                measurement_valid = False,
                normalized_aperture = None,
                fully_open = False,
                fully_closed = False,
                pinch_ratio = pinch_ratio,
                reasons = measurement_reasons,
            )

        assert pinch_ratio is not None

        aperture_unclamped = (pinch_ratio - self.config.closed_reference_ratio) / (
            self.config.open_reference_ratio - self.config.closed_reference_ratio)
        normalized_aperture = min(1.0, max(0.0, aperture_unclamped))

        return self._result(
            human,
            measurement_valid = True,
            normalized_aperture = normalized_aperture,
            fully_open = (normalized_aperture >= 1.0),
            fully_closed = (normalized_aperture <= 0.0),
            pinch_ratio = pinch_ratio,
            reasons = (),
        )

    def _result(
        self,
        human: HumanDerivedKinematics,
        *,
        measurement_valid: bool,
        normalized_aperture: float | None,
        fully_open: bool,
        fully_closed: bool,
        pinch_ratio: float | None,
        reasons: tuple[str, ...],
    ) -> GripperApertureResult:
        return GripperApertureResult(
            frame_sequence_id = human.frame_sequence_id,
            measurement_timestamp_s = human.measurement_timestamp_s,
            hand_side = self.hand_side,
            measurement_valid = measurement_valid,
            normalized_aperture = normalized_aperture,
            fully_open = fully_open,
            fully_closed = fully_closed,
            pinch_ratio = pinch_ratio,
            reasons = reasons,
        )

def _pinch_ratio_measurement( human: HumanDerivedKinematics, hand_side: HandSide) -> tuple[float | None, tuple[str, ...]]:
    hand = _selected_hand(human, hand_side)
    if hand is None:
        return None, (f"{hand_side.value}_hand_geometry_unavailable",)

    pinch_ratio = hand.pinch_ratio
    if not math.isfinite(pinch_ratio):
        return None, ("pinch_ratio_invalid",)
    if pinch_ratio < 0.0:
        return pinch_ratio, ("pinch_ratio_negative",)
    return pinch_ratio, ()

def _validate_human_order(
    human: HumanDerivedKinematics,
    last_frame_sequence_id: int | None,
    last_timestamp_s: float | None,
    *,
    stream_name: str,
) -> None:
    if human.frame_sequence_id < 0:
        raise ValueError("Human frame sequence ID must be non-negative.")
    if not math.isfinite(human.measurement_timestamp_s):
        raise ValueError("Human measurement timestamp must be finite.")
    if human.measurement_timestamp_s < 0.0:
        raise ValueError("Human measurement timestamp must be non-negative.")
    if (last_frame_sequence_id is not None and human.frame_sequence_id <= last_frame_sequence_id):
        raise ValueError(f"{stream_name} frame sequence IDs must be strictly increasing.")
    if (last_timestamp_s is not None and human.measurement_timestamp_s <= last_timestamp_s):
        raise ValueError(f"{stream_name} timestamps must be strictly increasing.")

def _selected_hand(human: HumanDerivedKinematics, hand_side: HandSide) -> HandDerivedKinematics | None:
    if hand_side == HandSide.LEFT:
        return human.left_hand
    if hand_side == HandSide.RIGHT:
        return human.right_hand
    raise ValueError(f"Unsupported hand side: {hand_side}")