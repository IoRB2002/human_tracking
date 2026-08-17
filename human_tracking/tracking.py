from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from human_tracking.observation import (
    HumanObservation,
    Landmark,
    LandmarkSet,
    ObservationValidity,
)


class TrackingState(str, Enum):
    """
    Temporal state of one tracked body/hand channel.
    """

    UNSEEN = "unseen"
    ACQUIRING = "acquiring"
    TRACKING = "tracking"
    DROPOUT = "dropout"
    LOST = "lost"


@dataclass(frozen=True)
class TemporalTrackingConfig:
    """
    Preliminary temporal validity parameters.

    These values are configurable and are not yet final thesis thresholds.
    """

    consecutive_valid_required: int = 3
    dropout_timeout_s: float = 0.25

    def __post_init__(self) -> None:
        if self.consecutive_valid_required < 1:
            raise ValueError(
                "consecutive_valid_required must be at least 1."
            )

        if not math.isfinite(
            self.dropout_timeout_s
        ):
            raise ValueError(
                "dropout_timeout_s must be finite."
            )

        if self.dropout_timeout_s < 0.0:
            raise ValueError(
                "dropout_timeout_s must be non-negative."
            )


@dataclass(frozen=True)
class ChannelTrackingResult:
    """
    Temporal result for body, left hand, or right hand.
    """

    state: TrackingState

    current_frame_valid: bool
    valid_for_control: bool

    consecutive_valid_frames: int

    age_since_last_valid_s: float | None

    reasons: tuple[str, ...]


@dataclass(frozen=True)
class HumanTrackingResult:
    """
    Transport-independent temporal result for one HumanObservation.
    """

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
    """
    Stateful temporal gate operating on HumanObservation validity.

    Important:
    - a dropout does NOT remain control-valid;
    - the timeout only distinguishes short dropout from tracking loss;
    - reacquisition requires consecutive valid observations;
    - smoothing is intentionally performed by HumanLandmarkSmoother below.
    """

    def __init__(
        self,
        config: TemporalTrackingConfig,
    ):
        self.config = config

        self._body = _ChannelMemory()
        self._left_hand = _ChannelMemory()
        self._right_hand = _ChannelMemory()

        self._last_sequence_id: int | None = None
        self._last_timestamp_s: float | None = None

    def update(
        self,
        observation: HumanObservation,
        validity: ObservationValidity,
    ) -> HumanTrackingResult:
        self._validate_order(observation)

        timestamp_s = (
            observation.measurement_timestamp_s
        )

        body_result = self._update_channel(
            memory=self._body,
            current_frame_valid=(
                validity.body_valid_for_control
            ),
            current_reasons=(
                validity.body_reasons
            ),
            timestamp_s=timestamp_s,
        )

        left_result = self._update_channel(
            memory=self._left_hand,
            current_frame_valid=(
                validity.left_hand_valid_for_control
            ),
            current_reasons=(
                validity.left_hand_reasons
            ),
            timestamp_s=timestamp_s,
        )

        right_result = self._update_channel(
            memory=self._right_hand,
            current_frame_valid=(
                validity.right_hand_valid_for_control
            ),
            current_reasons=(
                validity.right_hand_reasons
            ),
            timestamp_s=timestamp_s,
        )

        self._last_sequence_id = (
            observation.frame_sequence_id
        )

        self._last_timestamp_s = (
            observation.measurement_timestamp_s
        )

        return HumanTrackingResult(
            frame_sequence_id=(
                observation.frame_sequence_id
            ),
            measurement_timestamp_s=(
                observation.measurement_timestamp_s
            ),
            available_for_visualization=(
                validity.available_for_visualization
            ),
            body=body_result,
            left_hand=left_result,
            right_hand=right_result,
        )

    def _validate_order(
        self,
        observation: HumanObservation,
    ) -> None:
        timestamp_s = (
            observation.measurement_timestamp_s
        )

        if not math.isfinite(timestamp_s):
            raise ValueError(
                "Observation timestamp must be finite."
            )

        if timestamp_s < 0.0:
            raise ValueError(
                "Observation timestamp must be non-negative."
            )

        if self._last_sequence_id is not None:
            if (
                observation.frame_sequence_id
                <= self._last_sequence_id
            ):
                raise ValueError(
                    "Observation sequence IDs must "
                    "be strictly increasing."
                )

        if self._last_timestamp_s is not None:
            if (
                timestamp_s
                <= self._last_timestamp_s
            ):
                raise ValueError(
                    "Observation timestamps must "
                    "be strictly increasing."
                )

    def _update_channel(
        self,
        memory: _ChannelMemory,
        current_frame_valid: bool,
        current_reasons: tuple[str, ...],
        timestamp_s: float,
    ) -> ChannelTrackingResult:
        if current_frame_valid:
            return self._update_valid_channel(
                memory=memory,
                timestamp_s=timestamp_s,
            )

        return self._update_invalid_channel(
            memory=memory,
            timestamp_s=timestamp_s,
            current_reasons=current_reasons,
        )

    def _update_valid_channel(
        self,
        memory: _ChannelMemory,
        timestamp_s: float,
    ) -> ChannelTrackingResult:
        memory.last_valid_timestamp_s = (
            timestamp_s
        )

        if (
            memory.state
            == TrackingState.TRACKING
        ):
            memory.consecutive_valid_frames = (
                self.config.consecutive_valid_required
            )

            return ChannelTrackingResult(
                state=TrackingState.TRACKING,
                current_frame_valid=True,
                valid_for_control=True,
                consecutive_valid_frames=(
                    memory.consecutive_valid_frames
                ),
                age_since_last_valid_s=0.0,
                reasons=(),
            )

        memory.consecutive_valid_frames += 1

        if (
            memory.consecutive_valid_frames
            >= self.config.consecutive_valid_required
        ):
            memory.state = TrackingState.TRACKING
            memory.has_ever_tracked = True

            memory.consecutive_valid_frames = (
                self.config.consecutive_valid_required
            )

            return ChannelTrackingResult(
                state=TrackingState.TRACKING,
                current_frame_valid=True,
                valid_for_control=True,
                consecutive_valid_frames=(
                    memory.consecutive_valid_frames
                ),
                age_since_last_valid_s=0.0,
                reasons=(),
            )

        memory.state = TrackingState.ACQUIRING

        return ChannelTrackingResult(
            state=TrackingState.ACQUIRING,
            current_frame_valid=True,
            valid_for_control=False,
            consecutive_valid_frames=(
                memory.consecutive_valid_frames
            ),
            age_since_last_valid_s=0.0,
            reasons=(
                "temporal_acquiring",
            ),
        )

    def _update_invalid_channel(
        self,
        memory: _ChannelMemory,
        timestamp_s: float,
        current_reasons: tuple[str, ...],
    ) -> ChannelTrackingResult:
        memory.consecutive_valid_frames = 0

        if not memory.has_ever_tracked:
            memory.state = TrackingState.UNSEEN

            return ChannelTrackingResult(
                state=TrackingState.UNSEEN,
                current_frame_valid=False,
                valid_for_control=False,
                consecutive_valid_frames=0,
                age_since_last_valid_s=None,
                reasons=(
                    tuple(current_reasons)
                    + ("temporal_unseen",)
                ),
            )

        if memory.last_valid_timestamp_s is None:
            raise RuntimeError(
                "Temporal tracker reached an "
                "inconsistent internal state."
            )

        age_s = (
            timestamp_s
            - memory.last_valid_timestamp_s
        )

        if age_s < 0.0:
            raise RuntimeError(
                "Computed a negative measurement age."
            )

        if (
            age_s
            <= self.config.dropout_timeout_s
        ):
            memory.state = TrackingState.DROPOUT

            temporal_reason = (
                "temporal_dropout"
            )
        else:
            memory.state = TrackingState.LOST

            temporal_reason = (
                "temporal_lost"
            )

        return ChannelTrackingResult(
            state=memory.state,
            current_frame_valid=False,
            valid_for_control=False,
            consecutive_valid_frames=0,
            age_since_last_valid_s=age_s,
            reasons=(
                tuple(current_reasons)
                + (temporal_reason,)
            ),
        )


@dataclass(frozen=True)
class LandmarkSmoothingConfig:
    """
    Timestamp-aware first-order low-pass smoothing.

    time_constant_s is preliminary and will later be tuned experimentally.
    """

    time_constant_s: float = 0.10

    def __post_init__(self) -> None:
        if not math.isfinite(
            self.time_constant_s
        ):
            raise ValueError(
                "time_constant_s must be finite."
            )

        if self.time_constant_s <= 0.0:
            raise ValueError(
                "time_constant_s must be greater than zero."
            )


@dataclass(frozen=True)
class SmoothedChannelLandmarks:
    """
    Smoothed landmarks for one control channel.

    None means the temporal gate did not consider the channel control-valid.
    """

    image_landmarks: LandmarkSet | None
    world_landmarks: LandmarkSet | None


@dataclass(frozen=True)
class SmoothedHumanLandmarks:
    """
    Smoothed control-side landmark streams for one observation.

    This contains no robot mapping and does not convert MediaPipe world
    coordinates into workspace coordinates.
    """

    frame_sequence_id: int
    measurement_timestamp_s: float

    body: SmoothedChannelLandmarks
    left_hand: SmoothedChannelLandmarks
    right_hand: SmoothedChannelLandmarks


class _LandmarkSetFilter:
    def __init__(
        self,
        config: LandmarkSmoothingConfig,
    ):
        self.config = config
        self._previous: LandmarkSet | None = None
        self._previous_timestamp_s: float | None = None

    def reset(self) -> None:
        self._previous = None
        self._previous_timestamp_s = None

    def update(
        self,
        landmark_set: LandmarkSet,
        timestamp_s: float,
    ) -> LandmarkSet:
        if not math.isfinite(timestamp_s):
            raise ValueError(
                "Smoothing timestamp must be finite."
            )

        if self._previous is None:
            self._previous = landmark_set
            self._previous_timestamp_s = timestamp_s
            return landmark_set

        assert self._previous_timestamp_s is not None

        dt_s = (
            timestamp_s
            - self._previous_timestamp_s
        )

        if dt_s <= 0.0:
            raise ValueError(
                "Smoothing timestamps must be "
                "strictly increasing."
            )

        previous = self._previous

        if (
            previous.coordinate_space
            != landmark_set.coordinate_space
        ):
            raise ValueError(
                "Coordinate space changed inside "
                "one smoothing stream."
            )

        if (
            len(previous.landmarks)
            != len(landmark_set.landmarks)
        ):
            raise ValueError(
                "Landmark count changed inside "
                "one smoothing stream."
            )

        alpha = (
            1.0
            - math.exp(
                -dt_s
                / self.config.time_constant_s
            )
        )

        smoothed_landmarks = []

        for previous_landmark, current_landmark in zip(
            previous.landmarks,
            landmark_set.landmarks,
        ):
            if (
                previous_landmark.index
                != current_landmark.index
                or previous_landmark.name
                != current_landmark.name
            ):
                raise ValueError(
                    "Landmark identity changed inside "
                    "one smoothing stream."
                )

            smoothed_landmarks.append(
                Landmark(
                    index=current_landmark.index,
                    name=current_landmark.name,
                    x=(
                        previous_landmark.x
                        + alpha
                        * (
                            current_landmark.x
                            - previous_landmark.x
                        )
                    ),
                    y=(
                        previous_landmark.y
                        + alpha
                        * (
                            current_landmark.y
                            - previous_landmark.y
                        )
                    ),
                    z=(
                        previous_landmark.z
                        + alpha
                        * (
                            current_landmark.z
                            - previous_landmark.z
                        )
                    ),
                    visibility=(
                        current_landmark.visibility
                    ),
                    presence=(
                        current_landmark.presence
                    ),
                )
            )

        result = LandmarkSet(
            coordinate_space=(
                landmark_set.coordinate_space
            ),
            landmarks=tuple(
                smoothed_landmarks
            ),
        )

        self._previous = result
        self._previous_timestamp_s = timestamp_s

        return result


class HumanLandmarkSmoother:
    """
    Smooth only channels that have already passed temporal gating.

    If a channel becomes ACQUIRING, DROPOUT, LOST, or UNSEEN, its smoothing
    state is reset. Reacquisition therefore starts from fresh measurements
    rather than blending new data with stale coordinates.
    """

    def __init__(
        self,
        config: LandmarkSmoothingConfig,
    ):
        self.config = config

        self._body_image = _LandmarkSetFilter(
            config
        )
        self._body_world = _LandmarkSetFilter(
            config
        )

        self._left_image = _LandmarkSetFilter(
            config
        )
        self._left_world = _LandmarkSetFilter(
            config
        )

        self._right_image = _LandmarkSetFilter(
            config
        )
        self._right_world = _LandmarkSetFilter(
            config
        )

        self._last_sequence_id: int | None = None
        self._last_timestamp_s: float | None = None

    def update(
        self,
        observation: HumanObservation,
        tracking: HumanTrackingResult,
    ) -> SmoothedHumanLandmarks:
        self._validate_inputs(
            observation,
            tracking,
        )

        body = self._update_body(
            observation,
            tracking,
        )

        left_hand = self._update_hand(
            hands=observation.left_hands,
            channel=tracking.left_hand,
            image_filter=self._left_image,
            world_filter=self._left_world,
            side_name="left",
            timestamp_s=(
                observation.measurement_timestamp_s
            ),
        )

        right_hand = self._update_hand(
            hands=observation.right_hands,
            channel=tracking.right_hand,
            image_filter=self._right_image,
            world_filter=self._right_world,
            side_name="right",
            timestamp_s=(
                observation.measurement_timestamp_s
            ),
        )

        self._last_sequence_id = (
            observation.frame_sequence_id
        )

        self._last_timestamp_s = (
            observation.measurement_timestamp_s
        )

        return SmoothedHumanLandmarks(
            frame_sequence_id=(
                observation.frame_sequence_id
            ),
            measurement_timestamp_s=(
                observation.measurement_timestamp_s
            ),
            body=body,
            left_hand=left_hand,
            right_hand=right_hand,
        )

    def _validate_inputs(
        self,
        observation: HumanObservation,
        tracking: HumanTrackingResult,
    ) -> None:
        if (
            observation.frame_sequence_id
            != tracking.frame_sequence_id
        ):
            raise ValueError(
                "Observation and tracking sequence IDs "
                "do not match."
            )

        if (
            observation.measurement_timestamp_s
            != tracking.measurement_timestamp_s
        ):
            raise ValueError(
                "Observation and tracking timestamps "
                "do not match."
            )

        if self._last_sequence_id is not None:
            if (
                observation.frame_sequence_id
                <= self._last_sequence_id
            ):
                raise ValueError(
                    "Smoother sequence IDs must "
                    "be strictly increasing."
                )

        if self._last_timestamp_s is not None:
            if (
                observation.measurement_timestamp_s
                <= self._last_timestamp_s
            ):
                raise ValueError(
                    "Smoother timestamps must "
                    "be strictly increasing."
                )

    def _update_body(
        self,
        observation: HumanObservation,
        tracking: HumanTrackingResult,
    ) -> SmoothedChannelLandmarks:
        if not tracking.body.valid_for_control:
            self._body_image.reset()
            self._body_world.reset()

            return SmoothedChannelLandmarks(
                image_landmarks=None,
                world_landmarks=None,
            )

        if observation.body_image_landmarks is None:
            raise ValueError(
                "Body is control-valid but image "
                "landmarks are missing."
            )

        timestamp_s = (
            observation.measurement_timestamp_s
        )

        image_landmarks = (
            self._body_image.update(
                observation.body_image_landmarks,
                timestamp_s,
            )
        )

        if observation.body_world_landmarks is None:
            self._body_world.reset()
            world_landmarks = None
        else:
            world_landmarks = (
                self._body_world.update(
                    observation.body_world_landmarks,
                    timestamp_s,
                )
            )

        return SmoothedChannelLandmarks(
            image_landmarks=image_landmarks,
            world_landmarks=world_landmarks,
        )

    @staticmethod
    def _update_hand(
        hands,
        channel: ChannelTrackingResult,
        image_filter: _LandmarkSetFilter,
        world_filter: _LandmarkSetFilter,
        side_name: str,
        timestamp_s: float,
    ) -> SmoothedChannelLandmarks:
        if not channel.valid_for_control:
            image_filter.reset()
            world_filter.reset()

            return SmoothedChannelLandmarks(
                image_landmarks=None,
                world_landmarks=None,
            )

        if len(hands) != 1:
            raise ValueError(
                f"{side_name} hand is control-valid "
                "but does not contain exactly one detection."
            )

        hand = hands[0]

        image_landmarks = (
            image_filter.update(
                hand.image_landmarks,
                timestamp_s,
            )
        )

        if hand.world_landmarks is None:
            world_filter.reset()
            world_landmarks = None
        else:
            world_landmarks = (
                world_filter.update(
                    hand.world_landmarks,
                    timestamp_s,
                )
            )

        return SmoothedChannelLandmarks(
            image_landmarks=image_landmarks,
            world_landmarks=world_landmarks,
        )

@dataclass(frozen=True)
class Vector3:
    """
    Small transport-independent 3D vector type used by H6 geometry.
    """

    x: float
    y: float
    z: float


@dataclass(frozen=True)
class DerivedGeometryConfig:
    """
    Numerical guard for derived human geometry.

    This is not a workspace-distance threshold. It only prevents divisions
    by zero and unstable normalization in MediaPipe model-world coordinates.
    """

    min_length_model_world: float = 1e-5

    def __post_init__(self) -> None:
        if not math.isfinite(
            self.min_length_model_world
        ):
            raise ValueError(
                "min_length_model_world must be finite."
            )

        if self.min_length_model_world <= 0.0:
            raise ValueError(
                "min_length_model_world must be greater than zero."
            )


@dataclass(frozen=True)
class BodyRelativeFrame:
    """
    Orthonormal torso frame derived from smoothed body landmarks.

    right_axis points from anatomical left shoulder toward right shoulder.
    up_axis follows the torso from hip midpoint toward shoulder midpoint.
    normal_axis completes a right-handed orthonormal frame.

    The origin and axes are derived from MediaPipe model-world coordinates;
    they are not the Part 1 metric workspace frame.
    """

    origin_model_world: Vector3

    right_axis_model_world: Vector3
    up_axis_model_world: Vector3
    normal_axis_model_world: Vector3


@dataclass(frozen=True)
class BodyDerivedKinematics:
    """
    Body-relative arm geometry derived from one smoothed body sample.
    """

    frame: BodyRelativeFrame

    shoulder_width_model_world: float
    torso_height_model_world: float

    left_arm_length_model_world: float
    right_arm_length_model_world: float

    left_upper_arm_direction_body: Vector3
    left_forearm_direction_body: Vector3
    right_upper_arm_direction_body: Vector3
    right_forearm_direction_body: Vector3

    left_wrist_displacement_normalized_body: Vector3
    right_wrist_displacement_normalized_body: Vector3


@dataclass(frozen=True)
class HandDerivedKinematics:
    """
    Scale-normalized hand geometry.

    pinch_ratio is thumb-tip to index-tip distance divided by palm width.
    No discrete gesture or robot command is produced here.
    """

    palm_width_model_world: float
    pinch_ratio: float

    index_direction_model_world: Vector3
    palm_normal_model_world: Vector3


@dataclass(frozen=True)
class HumanDerivedKinematics:
    """
    H6 derived geometry for one temporally valid smoothed observation.
    """

    frame_sequence_id: int
    measurement_timestamp_s: float

    body: BodyDerivedKinematics | None
    left_hand: HandDerivedKinematics | None
    right_hand: HandDerivedKinematics | None

    body_reasons: tuple[str, ...]
    left_hand_reasons: tuple[str, ...]
    right_hand_reasons: tuple[str, ...]


def derive_human_kinematics(
    smoothed: SmoothedHumanLandmarks,
    config: DerivedGeometryConfig,
) -> HumanDerivedKinematics:
    """
    Derive robot-independent body and hand geometry.

    Only smoothed control-side streams are accepted. Missing or degenerate
    geometry becomes explicit None + diagnostic reasons rather than a zero
    vector or a fabricated measurement.
    """

    body, body_reasons = _derive_body_kinematics(
        smoothed.body.world_landmarks,
        config,
    )

    left_hand, left_reasons = _derive_hand_kinematics(
        smoothed.left_hand.world_landmarks,
        config,
        side_name="left",
    )

    right_hand, right_reasons = _derive_hand_kinematics(
        smoothed.right_hand.world_landmarks,
        config,
        side_name="right",
    )

    return HumanDerivedKinematics(
        frame_sequence_id=(
            smoothed.frame_sequence_id
        ),
        measurement_timestamp_s=(
            smoothed.measurement_timestamp_s
        ),
        body=body,
        left_hand=left_hand,
        right_hand=right_hand,
        body_reasons=tuple(body_reasons),
        left_hand_reasons=tuple(left_reasons),
        right_hand_reasons=tuple(right_reasons),
    )


def _derive_body_kinematics(
    landmark_set: LandmarkSet | None,
    config: DerivedGeometryConfig,
) -> tuple[
    BodyDerivedKinematics | None,
    list[str],
]:
    if landmark_set is None:
        return (
            None,
            ["body_world_landmarks_unavailable"],
        )

    if len(landmark_set.landmarks) != 33:
        return (
            None,
            ["body_world_landmark_count_invalid"],
        )

    landmarks = landmark_set.landmarks

    left_shoulder = _landmark_vector(
        landmarks[11]
    )
    right_shoulder = _landmark_vector(
        landmarks[12]
    )
    left_elbow = _landmark_vector(
        landmarks[13]
    )
    right_elbow = _landmark_vector(
        landmarks[14]
    )
    left_wrist = _landmark_vector(
        landmarks[15]
    )
    right_wrist = _landmark_vector(
        landmarks[16]
    )
    left_hip = _landmark_vector(
        landmarks[23]
    )
    right_hip = _landmark_vector(
        landmarks[24]
    )

    shoulder_midpoint = _scale_vector(
        _add_vectors(
            left_shoulder,
            right_shoulder,
        ),
        0.5,
    )

    hip_midpoint = _scale_vector(
        _add_vectors(
            left_hip,
            right_hip,
        ),
        0.5,
    )

    shoulder_vector = _subtract_vectors(
        right_shoulder,
        left_shoulder,
    )

    torso_vector = _subtract_vectors(
        shoulder_midpoint,
        hip_midpoint,
    )

    shoulder_width = _vector_norm(
        shoulder_vector
    )

    torso_height = _vector_norm(
        torso_vector
    )

    if (
        shoulder_width
        <= config.min_length_model_world
    ):
        return (
            None,
            ["body_shoulder_width_degenerate"],
        )

    if (
        torso_height
        <= config.min_length_model_world
    ):
        return (
            None,
            ["body_torso_height_degenerate"],
        )

    right_axis = _normalize_vector(
        shoulder_vector,
        config.min_length_model_world,
    )

    torso_up_seed = _normalize_vector(
        torso_vector,
        config.min_length_model_world,
    )

    normal_seed = _cross_vectors(
        right_axis,
        torso_up_seed,
    )

    if (
        _vector_norm(normal_seed)
        <= config.min_length_model_world
    ):
        return (
            None,
            ["body_frame_degenerate"],
        )

    normal_axis = _normalize_vector(
        normal_seed,
        config.min_length_model_world,
    )

    # Recompute up from the orthogonal axes so the final frame is
    # orthonormal even when the original torso vector is slightly skewed.
    up_axis = _normalize_vector(
        _cross_vectors(
            normal_axis,
            right_axis,
        ),
        config.min_length_model_world,
    )

    left_upper_world = _subtract_vectors(
        left_elbow,
        left_shoulder,
    )
    left_forearm_world = _subtract_vectors(
        left_wrist,
        left_elbow,
    )

    right_upper_world = _subtract_vectors(
        right_elbow,
        right_shoulder,
    )
    right_forearm_world = _subtract_vectors(
        right_wrist,
        right_elbow,
    )

    left_upper_length = _vector_norm(
        left_upper_world
    )
    left_forearm_length = _vector_norm(
        left_forearm_world
    )
    right_upper_length = _vector_norm(
        right_upper_world
    )
    right_forearm_length = _vector_norm(
        right_forearm_world
    )

    segment_lengths = (
        (
            "left_upper_arm",
            left_upper_length,
        ),
        (
            "left_forearm",
            left_forearm_length,
        ),
        (
            "right_upper_arm",
            right_upper_length,
        ),
        (
            "right_forearm",
            right_forearm_length,
        ),
    )

    for segment_name, length in segment_lengths:
        if (
            length
            <= config.min_length_model_world
        ):
            return (
                None,
                [
                    f"{segment_name}_length_degenerate"
                ],
            )

    left_arm_length = (
        left_upper_length
        + left_forearm_length
    )

    right_arm_length = (
        right_upper_length
        + right_forearm_length
    )

    left_upper_direction_body = (
        _vector_to_body_frame(
            _normalize_vector(
                left_upper_world,
                config.min_length_model_world,
            ),
            right_axis,
            up_axis,
            normal_axis,
        )
    )

    left_forearm_direction_body = (
        _vector_to_body_frame(
            _normalize_vector(
                left_forearm_world,
                config.min_length_model_world,
            ),
            right_axis,
            up_axis,
            normal_axis,
        )
    )

    right_upper_direction_body = (
        _vector_to_body_frame(
            _normalize_vector(
                right_upper_world,
                config.min_length_model_world,
            ),
            right_axis,
            up_axis,
            normal_axis,
        )
    )

    right_forearm_direction_body = (
        _vector_to_body_frame(
            _normalize_vector(
                right_forearm_world,
                config.min_length_model_world,
            ),
            right_axis,
            up_axis,
            normal_axis,
        )
    )

    left_wrist_from_shoulder = (
        _subtract_vectors(
            left_wrist,
            left_shoulder,
        )
    )

    right_wrist_from_shoulder = (
        _subtract_vectors(
            right_wrist,
            right_shoulder,
        )
    )

    left_wrist_normalized_world = (
        _scale_vector(
            left_wrist_from_shoulder,
            1.0 / left_arm_length,
        )
    )

    right_wrist_normalized_world = (
        _scale_vector(
            right_wrist_from_shoulder,
            1.0 / right_arm_length,
        )
    )

    frame = BodyRelativeFrame(
        origin_model_world=hip_midpoint,
        right_axis_model_world=right_axis,
        up_axis_model_world=up_axis,
        normal_axis_model_world=normal_axis,
    )

    return (
        BodyDerivedKinematics(
            frame=frame,
            shoulder_width_model_world=(
                shoulder_width
            ),
            torso_height_model_world=(
                torso_height
            ),
            left_arm_length_model_world=(
                left_arm_length
            ),
            right_arm_length_model_world=(
                right_arm_length
            ),
            left_upper_arm_direction_body=(
                left_upper_direction_body
            ),
            left_forearm_direction_body=(
                left_forearm_direction_body
            ),
            right_upper_arm_direction_body=(
                right_upper_direction_body
            ),
            right_forearm_direction_body=(
                right_forearm_direction_body
            ),
            left_wrist_displacement_normalized_body=(
                _vector_to_body_frame(
                    left_wrist_normalized_world,
                    right_axis,
                    up_axis,
                    normal_axis,
                )
            ),
            right_wrist_displacement_normalized_body=(
                _vector_to_body_frame(
                    right_wrist_normalized_world,
                    right_axis,
                    up_axis,
                    normal_axis,
                )
            ),
        ),
        [],
    )


def _derive_hand_kinematics(
    landmark_set: LandmarkSet | None,
    config: DerivedGeometryConfig,
    side_name: str,
) -> tuple[
    HandDerivedKinematics | None,
    list[str],
]:
    if landmark_set is None:
        return (
            None,
            [
                f"{side_name}_hand_world_landmarks_unavailable"
            ],
        )

    if len(landmark_set.landmarks) != 21:
        return (
            None,
            [
                f"{side_name}_hand_world_landmark_count_invalid"
            ],
        )

    landmarks = landmark_set.landmarks

    wrist = _landmark_vector(
        landmarks[0]
    )
    thumb_tip = _landmark_vector(
        landmarks[4]
    )
    index_mcp = _landmark_vector(
        landmarks[5]
    )
    index_tip = _landmark_vector(
        landmarks[8]
    )
    pinky_mcp = _landmark_vector(
        landmarks[17]
    )

    palm_width_vector = _subtract_vectors(
        index_mcp,
        pinky_mcp,
    )

    palm_width = _vector_norm(
        palm_width_vector
    )

    if (
        palm_width
        <= config.min_length_model_world
    ):
        return (
            None,
            [
                f"{side_name}_hand_palm_width_degenerate"
            ],
        )

    pinch_distance = _vector_norm(
        _subtract_vectors(
            thumb_tip,
            index_tip,
        )
    )

    index_vector = _subtract_vectors(
        index_tip,
        wrist,
    )

    if (
        _vector_norm(index_vector)
        <= config.min_length_model_world
    ):
        return (
            None,
            [
                f"{side_name}_hand_index_direction_degenerate"
            ],
        )

    wrist_to_index_mcp = (
        _subtract_vectors(
            index_mcp,
            wrist,
        )
    )

    wrist_to_pinky_mcp = (
        _subtract_vectors(
            pinky_mcp,
            wrist,
        )
    )

    palm_normal_seed = _cross_vectors(
        wrist_to_index_mcp,
        wrist_to_pinky_mcp,
    )

    if (
        _vector_norm(palm_normal_seed)
        <= config.min_length_model_world
    ):
        return (
            None,
            [
                f"{side_name}_hand_palm_normal_degenerate"
            ],
        )

    return (
        HandDerivedKinematics(
            palm_width_model_world=(
                palm_width
            ),
            pinch_ratio=(
                pinch_distance
                / palm_width
            ),
            index_direction_model_world=(
                _normalize_vector(
                    index_vector,
                    config.min_length_model_world,
                )
            ),
            palm_normal_model_world=(
                _normalize_vector(
                    palm_normal_seed,
                    config.min_length_model_world,
                )
            ),
        ),
        [],
    )


def _landmark_vector(
    landmark: Landmark,
) -> Vector3:
    values = (
        float(landmark.x),
        float(landmark.y),
        float(landmark.z),
    )

    if not all(
        math.isfinite(value)
        for value in values
    ):
        raise ValueError(
            f"Landmark {landmark.name} contains "
            "non-finite coordinates."
        )

    return Vector3(
        x=values[0],
        y=values[1],
        z=values[2],
    )


def _add_vectors(
    first: Vector3,
    second: Vector3,
) -> Vector3:
    return Vector3(
        x=first.x + second.x,
        y=first.y + second.y,
        z=first.z + second.z,
    )


def _subtract_vectors(
    first: Vector3,
    second: Vector3,
) -> Vector3:
    return Vector3(
        x=first.x - second.x,
        y=first.y - second.y,
        z=first.z - second.z,
    )


def _scale_vector(
    vector: Vector3,
    scale: float,
) -> Vector3:
    return Vector3(
        x=vector.x * scale,
        y=vector.y * scale,
        z=vector.z * scale,
    )


def _vector_norm(
    vector: Vector3,
) -> float:
    return math.sqrt(
        vector.x * vector.x
        + vector.y * vector.y
        + vector.z * vector.z
    )


def _normalize_vector(
    vector: Vector3,
    min_norm: float,
) -> Vector3:
    norm = _vector_norm(
        vector
    )

    if norm <= min_norm:
        raise ValueError(
            "Cannot normalize a degenerate vector."
        )

    return _scale_vector(
        vector,
        1.0 / norm,
    )


def _dot_vectors(
    first: Vector3,
    second: Vector3,
) -> float:
    return (
        first.x * second.x
        + first.y * second.y
        + first.z * second.z
    )


def _cross_vectors(
    first: Vector3,
    second: Vector3,
) -> Vector3:
    return Vector3(
        x=(
            first.y * second.z
            - first.z * second.y
        ),
        y=(
            first.z * second.x
            - first.x * second.z
        ),
        z=(
            first.x * second.y
            - first.y * second.x
        ),
    )


def _vector_to_body_frame(
    vector_model_world: Vector3,
    right_axis: Vector3,
    up_axis: Vector3,
    normal_axis: Vector3,
) -> Vector3:
    return Vector3(
        x=_dot_vectors(
            vector_model_world,
            right_axis,
        ),
        y=_dot_vectors(
            vector_model_world,
            up_axis,
        ),
        z=_dot_vectors(
            vector_model_world,
            normal_axis,
        ),
    )

