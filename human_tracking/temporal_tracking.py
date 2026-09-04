from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import math
from human_tracking.observation import HumanObservation, ObservationValidity

class TrackingState(str, Enum):
    UNSEEN = "unseen"
    ACQUIRING = "acquiring"
    TRACKING = "tracking"
    DROPOUT = "dropout"
    LOST = "lost"

@dataclass(frozen=True)
class TemporalTrackingConfig:
    consecutive_valid_required: int = 3
    dropout_timeout_s: float = 0.25

    def __post_init__(self) -> None:
        if self.consecutive_valid_required < 1:
            raise ValueError("consecutive_valid_required must be at least 1.")

        if not math.isfinite(self.dropout_timeout_s):
            raise ValueError("dropout_timeout_s must be finite.")

        if self.dropout_timeout_s < 0.0:
            raise ValueError("dropout_timeout_s must be non-negative.")

@dataclass(frozen=True)
class ChannelTrackingResult:
    state: TrackingState
    current_frame_valid: bool
    valid_for_control: bool
    consecutive_valid_frames: int
    age_since_last_valid_s: float | None
    reasons: tuple[str, ...]

@dataclass(frozen=True)
class HumanTrackingResult:
    frame_sequence_id: int
    measurement_timestamp_s: float
    available_for_visualization: bool
    body: ChannelTrackingResult
    left_hand: ChannelTrackingResult
    right_hand: ChannelTrackingResult

@dataclass
class _ChannelMemory:
    state: TrackingState = TrackingState.UNSEEN
    consecutive_valid_frames: int = 0
    last_valid_timestamp_s: float | None = None
    has_ever_tracked: bool = False

class HumanTemporalTracker:
    def __init__(self, config: TemporalTrackingConfig):
        self.config = config
        self._body = _ChannelMemory()
        self._left_hand = _ChannelMemory()
        self._right_hand = _ChannelMemory()
        self._last_sequence_id: int | None = None
        self._last_timestamp_s: float | None = None

    def update(self, observation: HumanObservation, validity: ObservationValidity) -> HumanTrackingResult:
        self._validate_order(observation)

        timestamp_s = observation.measurement_timestamp_s

        body_result = self._update_channel(
            memory = self._body,
            current_frame_valid = validity.body_valid_for_control,
            current_reasons = validity.body_reasons,
            timestamp_s = timestamp_s
        )

        left_result = self._update_channel(
            memory = self._left_hand,
            current_frame_valid = validity.left_hand_valid_for_control,
            current_reasons = validity.left_hand_reasons,
            timestamp_s = timestamp_s
        )

        right_result = self._update_channel(
            memory = self._right_hand,
            current_frame_valid = validity.right_hand_valid_for_control,
            current_reasons = validity.right_hand_reasons,
            timestamp_s = timestamp_s
        )

        self._last_sequence_id = observation.frame_sequence_id
        self._last_timestamp_s = observation.measurement_timestamp_s

        return HumanTrackingResult(
            frame_sequence_id = observation.frame_sequence_id,
            measurement_timestamp_s = observation.measurement_timestamp_s,
            available_for_visualization = validity.available_for_visualization,
            body = body_result,
            left_hand = left_result,
            right_hand = right_result
        )

    def _validate_order(self, observation: HumanObservation) -> None:
        timestamp_s = observation.measurement_timestamp_s

        if not math.isfinite(timestamp_s):
            raise ValueError("Observation timestamp must be finite.")

        if timestamp_s < 0.0:
            raise ValueError("Observation timestamp must be non-negative.")

        if self._last_sequence_id is not None:
            if (observation.frame_sequence_id <= self._last_sequence_id):
                raise ValueError("Observation sequence IDs must be strictly increasing.")

        if self._last_timestamp_s is not None:
            if (timestamp_s <= self._last_timestamp_s):
                raise ValueError("Observation timestamps must be strictly increasing.")

    def _update_channel(
        self,
        memory: _ChannelMemory,
        current_frame_valid: bool,
        current_reasons: tuple[str, ...],
        timestamp_s: float,
    ) -> ChannelTrackingResult:
        if current_frame_valid:
            return self._update_valid_channel(memory = memory, timestamp_s = timestamp_s)

        return self._update_invalid_channel(
            memory = memory,
            timestamp_s = timestamp_s,
            current_reasons = current_reasons
        )

    def _update_valid_channel(self, memory: _ChannelMemory, timestamp_s: float) -> ChannelTrackingResult:
        memory.last_valid_timestamp_s = (timestamp_s)

        if (memory.state == TrackingState.TRACKING):
            memory.consecutive_valid_frames = self.config.consecutive_valid_required

            return ChannelTrackingResult(
                state=TrackingState.TRACKING,
                current_frame_valid = True,
                valid_for_control = True,
                consecutive_valid_frames = memory.consecutive_valid_frames,
                age_since_last_valid_s = 0.0,
                reasons = ()
            )

        memory.consecutive_valid_frames += 1

        if memory.consecutive_valid_frames >= self.config.consecutive_valid_required:
            memory.state = TrackingState.TRACKING
            memory.has_ever_tracked = True
            memory.consecutive_valid_frames = self.config.consecutive_valid_required

            return ChannelTrackingResult(
                state=TrackingState.TRACKING,
                current_frame_valid = True,
                valid_for_control = True,
                consecutive_valid_frames = memory.consecutive_valid_frames,
                age_since_last_valid_s = 0.0,
                reasons = ()
            )

        memory.state = TrackingState.ACQUIRING

        return ChannelTrackingResult(
            state = TrackingState.ACQUIRING,
            current_frame_valid = True,
            valid_for_control = False,
            consecutive_valid_frames = memory.consecutive_valid_frames,
            age_since_last_valid_s = 0.0,
            reasons = ("temporal_acquiring",)
        )

    def _update_invalid_channel(self, memory: _ChannelMemory, timestamp_s: float, current_reasons: tuple[str, ...]) -> ChannelTrackingResult:
        memory.consecutive_valid_frames = 0

        if not memory.has_ever_tracked:
            memory.state = TrackingState.UNSEEN

            return ChannelTrackingResult(
                state = TrackingState.UNSEEN,
                current_frame_valid = False,
                valid_for_control = False,
                consecutive_valid_frames = 0,
                age_since_last_valid_s = None,
                reasons = tuple(current_reasons) + ("temporal_unseen",)
            )

        if memory.last_valid_timestamp_s is None:
            raise RuntimeError("Temporal tracker reached an inconsistent internal state.")

        age_s = timestamp_s - memory.last_valid_timestamp_s

        if age_s < 0.0:
            raise RuntimeError("Computed a negative measurement age.")

        if age_s <= self.config.dropout_timeout_s:
            memory.state = TrackingState.DROPOUT
            temporal_reason = "temporal_dropout"

        else:
            memory.state = TrackingState.LOST
            temporal_reason = "temporal_lost"

        return ChannelTrackingResult(
            state = memory.state,
            current_frame_valid = False,
            valid_for_control = False,
            consecutive_valid_frames = 0,
            age_since_last_valid_s = age_s,
            reasons = tuple(current_reasons) + (temporal_reason,)
        )