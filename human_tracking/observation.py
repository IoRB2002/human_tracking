from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from typing import Any


class CoordinateSpace(str, Enum):
    """
    Explicit coordinate-space labels used by human observations.

    MediaPipe world coordinates are deliberately not called workspace
    coordinates. They are model-provided coordinates and require separate
    interpretation before any robot mapping is attempted.
    """

    NORMALIZED_IMAGE = "normalized_image"
    MEDIAPIPE_WORLD = "mediapipe_world"


@dataclass(frozen=True)
class Landmark:
    """
    One landmark expressed in a named coordinate space.
    """

    index: int
    name: str

    x: float
    y: float
    z: float

    visibility: float | None = None
    presence: float | None = None


@dataclass(frozen=True)
class LandmarkSet:
    """
    A complete set of landmarks expressed in one coordinate space.
    """

    coordinate_space: CoordinateSpace
    landmarks: tuple[Landmark, ...]

    def __len__(self) -> int:
        return len(self.landmarks)


@dataclass(frozen=True)
class HandObservation:
    """
    One detected hand.

    handedness is the backend-provided classification. Identity continuity
    across frames will be handled later by the temporal tracking layer.
    """

    handedness: str
    handedness_score: float | None

    image_landmarks: LandmarkSet
    world_landmarks: LandmarkSet | None


@dataclass(frozen=True)
class HumanObservation:
    """
    Transport-independent observation derived from one camera image.

    This structure describes what was observed. It does not itself mean that
    the observation is valid for robot control.
    """

    frame_sequence_id: int
    measurement_timestamp_s: float

    image_width_px: int
    image_height_px: int

    body_image_landmarks: LandmarkSet | None
    body_world_landmarks: LandmarkSet | None

    left_hands: tuple[HandObservation, ...]
    right_hands: tuple[HandObservation, ...]
    unknown_hands: tuple[HandObservation, ...]

    backend_name: str
    backend_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_sequence_id": self.frame_sequence_id,
            "measurement_timestamp_s": self.measurement_timestamp_s,
            "image_width_px": self.image_width_px,
            "image_height_px": self.image_height_px,
            "body_image_landmarks": _landmark_set_to_dict(
                self.body_image_landmarks
            ),
            "body_world_landmarks": _landmark_set_to_dict(
                self.body_world_landmarks
            ),
            "left_hands": [
                _hand_to_dict(hand)
                for hand in self.left_hands
            ],
            "right_hands": [
                _hand_to_dict(hand)
                for hand in self.right_hands
            ],
            "unknown_hands": [
                _hand_to_dict(hand)
                for hand in self.unknown_hands
            ],
            "backend_name": self.backend_name,
            "backend_version": self.backend_version,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            separators=(",", ":"),
        )


@dataclass(frozen=True)
class ObservationValidationConfig:
    """
    Tunable per-frame quality thresholds.

    These are not final thesis thresholds. They are initial configurable
    gates that can later be justified experimentally.
    """

    min_body_visibility: float = 0.50
    min_body_presence: float = 0.50
    min_handedness_score: float = 0.50

    def __post_init__(self) -> None:
        for name, value in (
            (
                "min_body_visibility",
                self.min_body_visibility,
            ),
            (
                "min_body_presence",
                self.min_body_presence,
            ),
            (
                "min_handedness_score",
                self.min_handedness_score,
            ),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be between 0.0 and 1.0."
                )


@dataclass(frozen=True)
class ObservationValidity:
    """
    Stateless quality assessment of one HumanObservation.

    Temporal dropout, identity continuity, reacquisition, and smoothing are
    intentionally excluded from this structure.
    """

    available_for_visualization: bool

    body_valid_for_control: bool
    left_hand_valid_for_control: bool
    right_hand_valid_for_control: bool

    body_reasons: tuple[str, ...]
    left_hand_reasons: tuple[str, ...]
    right_hand_reasons: tuple[str, ...]


# The body representation required for later arm/torso processing.
_REQUIRED_BODY_LANDMARK_INDICES = (
    11,  # left shoulder
    12,  # right shoulder
    13,  # left elbow
    14,  # right elbow
    15,  # left wrist
    16,  # right wrist
    23,  # left hip
    24,  # right hip
)


def evaluate_observation(
    observation: HumanObservation,
    config: ObservationValidationConfig,
) -> ObservationValidity:
    """
    Evaluate one observation without using previous or future frames.

    The purpose of H4 is to distinguish:
      - data that may be displayed;
      - body data acceptable for later control calculations;
      - left-hand data acceptable for later gesture calculations;
      - right-hand data acceptable for later gesture calculations.
    """

    available_for_visualization = (
        observation.body_image_landmarks is not None
        or bool(observation.left_hands)
        or bool(observation.right_hands)
        or bool(observation.unknown_hands)
    )

    body_reasons = _validate_body(
        observation,
        config,
    )

    left_hand_reasons = _validate_hand_side(
        hands=observation.left_hands,
        expected_handedness="Left",
        config=config,
    )

    right_hand_reasons = _validate_hand_side(
        hands=observation.right_hands,
        expected_handedness="Right",
        config=config,
    )

    return ObservationValidity(
        available_for_visualization=(
            available_for_visualization
        ),
        body_valid_for_control=(
            len(body_reasons) == 0
        ),
        left_hand_valid_for_control=(
            len(left_hand_reasons) == 0
        ),
        right_hand_valid_for_control=(
            len(right_hand_reasons) == 0
        ),
        body_reasons=tuple(body_reasons),
        left_hand_reasons=tuple(
            left_hand_reasons
        ),
        right_hand_reasons=tuple(
            right_hand_reasons
        ),
    )


def _validate_body(
    observation: HumanObservation,
    config: ObservationValidationConfig,
) -> list[str]:
    reasons: list[str] = []

    landmark_set = (
        observation.body_image_landmarks
    )

    if landmark_set is None:
        return ["body_missing"]

    if len(landmark_set.landmarks) != 33:
        return ["body_landmark_count_invalid"]

    for index in _REQUIRED_BODY_LANDMARK_INDICES:
        landmark = landmark_set.landmarks[index]

        if landmark.visibility is None:
            reasons.append(
                f"{landmark.name}_visibility_missing"
            )
        elif (
            landmark.visibility
            < config.min_body_visibility
        ):
            reasons.append(
                f"{landmark.name}_visibility_low"
            )

        if landmark.presence is None:
            reasons.append(
                f"{landmark.name}_presence_missing"
            )
        elif (
            landmark.presence
            < config.min_body_presence
        ):
            reasons.append(
                f"{landmark.name}_presence_low"
            )

    return reasons


def _validate_hand_side(
    hands: tuple[HandObservation, ...],
    expected_handedness: str,
    config: ObservationValidationConfig,
) -> list[str]:
    reasons: list[str] = []

    side = expected_handedness.lower()

    if len(hands) == 0:
        return [f"{side}_hand_missing"]

    if len(hands) > 1:
        return [f"{side}_hand_ambiguous"]

    hand = hands[0]

    if (
        len(hand.image_landmarks.landmarks)
        != 21
    ):
        reasons.append(
            f"{side}_hand_landmark_count_invalid"
        )

    if (
        hand.handedness.strip().lower()
        != side
    ):
        reasons.append(
            f"{side}_hand_classification_mismatch"
        )

    if hand.handedness_score is None:
        reasons.append(
            f"{side}_handedness_score_missing"
        )
    elif (
        hand.handedness_score
        < config.min_handedness_score
    ):
        reasons.append(
            f"{side}_handedness_score_low"
        )

    return reasons


def _landmark_to_dict(
    landmark: Landmark,
) -> dict[str, Any]:
    return {
        "index": landmark.index,
        "name": landmark.name,
        "x": landmark.x,
        "y": landmark.y,
        "z": landmark.z,
        "visibility": landmark.visibility,
        "presence": landmark.presence,
    }


def _landmark_set_to_dict(
    landmark_set: LandmarkSet | None,
) -> dict[str, Any] | None:
    if landmark_set is None:
        return None

    return {
        "coordinate_space": (
            landmark_set.coordinate_space.value
        ),
        "landmarks": [
            _landmark_to_dict(landmark)
            for landmark in landmark_set.landmarks
        ],
    }


def _hand_to_dict(
    hand: HandObservation,
) -> dict[str, Any]:
    return {
        "handedness": hand.handedness,
        "handedness_score": hand.handedness_score,
        "image_landmarks": _landmark_set_to_dict(
            hand.image_landmarks
        ),
        "world_landmarks": _landmark_set_to_dict(
            hand.world_landmarks
        ),
    }