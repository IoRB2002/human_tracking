from __future__ import annotations
from dataclasses import dataclass, replace
from enum import Enum
import json
import math
from typing import Any

class CoordinateSpace(str, Enum):
    NORMALIZED_IMAGE = "normalized_image"
    MEDIAPIPE_WORLD = "mediapipe_world"

class HandAssociationSource(str, Enum):
    BACKEND_HANDEDNESS = "backend_handedness"
    POSE_WRIST = "pose_wrist"

class ControlArm(str, Enum):
    LEFT = "left"
    RIGHT = "right"

@dataclass(frozen=True)
class Landmark:
    index: int
    name: str
    x: float
    y: float
    z: float
    visibility: float | None = None
    presence: float | None = None

@dataclass(frozen=True)
class LandmarkSet:
    coordinate_space: CoordinateSpace
    landmarks: tuple[Landmark, ...]

    def __len__(self) -> int:
        return len(self.landmarks)

@dataclass(frozen=True)
class HandObservation:
    handedness: str
    handedness_score: float | None
    image_landmarks: LandmarkSet
    world_landmarks: LandmarkSet | None
    association_source: HandAssociationSource = HandAssociationSource.BACKEND_HANDEDNESS

@dataclass(frozen=True)
class HumanObservation:
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
            "body_image_landmarks": _landmark_set_to_dict(self.body_image_landmarks),
            "body_world_landmarks": _landmark_set_to_dict(self.body_world_landmarks),
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
            "backend_version": self.backend_version
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

@dataclass(frozen=True)
class ObservationValidationConfig:
    selected_control_arm: ControlArm
    min_body_visibility: float = 0.50
    min_body_presence: float = 0.50
    min_handedness_score: float = 0.50

    enable_pose_hand_association: bool = False
    max_hand_wrist_pose_distance_shoulder_widths: float = 0.50

    def __post_init__(self) -> None:
        try:
            selected_control_arm = ControlArm(self.selected_control_arm)

        except ValueError as exc:
            raise ValueError("selected_control_arm must be 'left' or 'right'.") from exc

        object.__setattr__(self, "selected_control_arm", selected_control_arm)

        for name, value in (
            ("min_body_visibility", self.min_body_visibility),
            ("min_body_presence", self.min_body_presence),
            ("min_handedness_score", self.min_handedness_score)
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0.")

        gate = self.max_hand_wrist_pose_distance_shoulder_widths
        if not math.isfinite(gate):
            raise ValueError("max_hand_wrist_pose_distance_shoulder_widths must be finite.")

        if gate <= 0.0:
            raise ValueError("max_hand_wrist_pose_distance_shoulder_widths must be positive.")

@dataclass(frozen=True)
class ObservationValidity:
    available_for_visualization: bool
    body_valid_for_control: bool
    left_hand_valid_for_control: bool
    right_hand_valid_for_control: bool
    body_reasons: tuple[str, ...]
    left_hand_reasons: tuple[str, ...]
    right_hand_reasons: tuple[str, ...]

# Torso landmarks are required for the body-relative control frame.
_REQUIRED_TORSO_LANDMARK_INDICES = (
    11,  # left shoulder
    12,  # right shoulder
    23,  # left hip
    24,  # right hip
)

_SELECTED_ARM_LANDMARK_INDICES = {
    ControlArm.LEFT: (
        13,  # left elbow
        15,  # left wrist
    ),
    ControlArm.RIGHT: (
        14,  # right elbow
        16,  # right wrist
    ),
}

def associate_hands_to_pose(observation: HumanObservation, config: ObservationValidationConfig) -> HumanObservation:
    if not config.enable_pose_hand_association:
        return observation

    body = observation.body_image_landmarks
    if body is None or len(body.landmarks) != 33:
        return observation

    landmarks = body.landmarks
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]

    if not (
        _landmark_is_confident(left_shoulder, config)
        and _landmark_is_confident(right_shoulder, config)
        and _landmark_xy_is_finite(left_shoulder)
        and _landmark_xy_is_finite(right_shoulder)
    ):
        return observation

    if observation.image_width_px <= 0 or observation.image_height_px <= 0:
        return observation

    shoulder_width = _image_distance_px(
        left_shoulder,
        right_shoulder,
        observation.image_width_px,
        observation.image_height_px
    )

    if not math.isfinite(shoulder_width) or shoulder_width <= 1e-12:
        return observation

    pose_wrist_by_side: dict[str, Landmark] = {}
    for side, index in (("left", 15), ("right", 16)):
        wrist = landmarks[index]
        if _landmark_is_confident(wrist, config) and _landmark_xy_is_finite(wrist):
            pose_wrist_by_side[side] = wrist

    if not pose_wrist_by_side:
        return observation

    max_distance = config.max_hand_wrist_pose_distance_shoulder_widths * shoulder_width

    raw_hands = (
        [("left", hand) for hand in observation.left_hands]
        + [("right", hand) for hand in observation.right_hands]
        + [("unknown", hand) for hand in observation.unknown_hands]
    )

    associated_candidates: dict[str, list[tuple[float, HandObservation]]] = {
        "left": [],
        "right": []
    }

    backend_fallback: dict[str, list[HandObservation]] = {
        "left": [],
        "right": []
    }

    unknown_hands: list[HandObservation] = []

    for original_side, hand in raw_hands:
        image_landmarks = hand.image_landmarks.landmarks
        if len(image_landmarks) != 21:
            _retain_without_association(
                original_side,
                hand,
                pose_wrist_by_side,
                backend_fallback,
                unknown_hands
            )
            continue

        hand_wrist = image_landmarks[0]
        if not _landmark_xy_is_finite(hand_wrist):
            _retain_without_association(
                original_side,
                hand,
                pose_wrist_by_side,
                backend_fallback,
                unknown_hands
            )
            continue

        distances = {
            side: _image_distance_px(
                hand_wrist,
                pose_wrist,
                observation.image_width_px,
                observation.image_height_px
            )
            for side, pose_wrist in pose_wrist_by_side.items()
        }
        nearest_side, nearest_distance = min(distances.items(), key=lambda item: item[1])

        if nearest_distance <= max_distance:
            associated_candidates[nearest_side].append(
                (
                    nearest_distance,
                    replace(hand, association_source=HandAssociationSource.POSE_WRIST)
                )
            )
            continue

        _retain_without_association(
            original_side,
            hand,
            pose_wrist_by_side,
            backend_fallback,
            unknown_hands
        )

    output_by_side: dict[str, tuple[HandObservation, ...]] = {}
    for side in ("left", "right"):
        if side in pose_wrist_by_side:
            selected, extras = _select_nearest_associated_hand(associated_candidates[side]            )
            output_by_side[side] = selected
            unknown_hands.extend(extras)
        else:
            output_by_side[side] = tuple(backend_fallback[side])

    return replace(
        observation,
        left_hands=output_by_side["left"],
        right_hands=output_by_side["right"],
        unknown_hands=tuple(unknown_hands)
    )

def _retain_without_association(
    original_side: str,
    hand: HandObservation,
    pose_wrist_by_side: dict[str, Landmark],
    backend_fallback: dict[str, list[HandObservation]],
    unknown_hands: list[HandObservation]
) -> None:
    if (
        original_side in backend_fallback
        and original_side not in pose_wrist_by_side
    ):
        backend_fallback[original_side].append(hand)
    else:
        unknown_hands.append(hand)

def _select_nearest_associated_hand(candidates: list[tuple[float, HandObservation]]) -> tuple[tuple[HandObservation, ...], list[HandObservation]]:
    if not candidates:
        return (), []

    ordered = sorted(candidates, key=lambda item: item[0])
    selected = ordered[0][1]
    extras = [hand for _, hand in ordered[1:]]
    return (selected,), extras


def _landmark_is_confident(landmark: Landmark, config: ObservationValidationConfig) -> bool:
    return (
        landmark.visibility is not None
        and landmark.presence is not None
        and landmark.visibility >= config.min_body_visibility
        and landmark.presence >= config.min_body_presence
    )

def _landmark_xy_is_finite(landmark: Landmark) -> bool:
    return math.isfinite(landmark.x) and math.isfinite(landmark.y)

def _image_distance_px(
    first: Landmark,
    second: Landmark,
    image_width_px: int,
    image_height_px: int
) -> float:
    return math.hypot((first.x - second.x) * image_width_px, (first.y - second.y) * image_height_px)

def evaluate_observation(observation: HumanObservation, config: ObservationValidationConfig) -> ObservationValidity:
    available_for_visualization = (
        observation.body_image_landmarks is not None
        or bool(observation.left_hands)
        or bool(observation.right_hands)
        or bool(observation.unknown_hands)
    )

    body_reasons = _validate_body(observation, config)
    left_hand_reasons = _validate_hand_side(
        hands = observation.left_hands,
        expected_handedness = "Left",
        config = config
    )

    right_hand_reasons = _validate_hand_side(
        hands = observation.right_hands,
        expected_handedness = "Right",
        config = config
    )

    return ObservationValidity(
        available_for_visualization = available_for_visualization,
        body_valid_for_control = len(body_reasons) == 0,
        left_hand_valid_for_control = len(left_hand_reasons) == 0,
        right_hand_valid_for_control = len(right_hand_reasons) == 0,
        body_reasons = tuple(body_reasons),
        left_hand_reasons = tuple(left_hand_reasons),
        right_hand_reasons = tuple(right_hand_reasons)
    )


def _validate_body(observation: HumanObservation, config: ObservationValidationConfig) -> list[str]:
    reasons: list[str] = []
    landmark_set = observation.body_image_landmarks

    if landmark_set is None:
        return ["body_missing"]

    if len(landmark_set.landmarks) != 33:
        return ["body_landmark_count_invalid"]

    required_indices = (_REQUIRED_TORSO_LANDMARK_INDICES + _SELECTED_ARM_LANDMARK_INDICES[config.selected_control_arm])

    for index in required_indices:
        landmark = landmark_set.landmarks[index]

        if landmark.visibility is None:
            reasons.append (f"{landmark.name}_visibility_missing")
        elif (landmark.visibility < config.min_body_visibility):
            reasons.append(f"{landmark.name}_visibility_low")

        if landmark.presence is None:
            reasons.append(f"{landmark.name}_presence_missing")
        elif (landmark.presence < config.min_body_presence):
            reasons.append(f"{landmark.name}_presence_low")

    return reasons

def _validate_hand_side(hands: tuple[HandObservation, ...], expected_handedness: str, config: ObservationValidationConfig) -> list[str]:
    reasons: list[str] = []
    side = expected_handedness.lower()

    if len(hands) == 0:
        return [f"{side}_hand_missing"]

    if len(hands) > 1:
        return [f"{side}_hand_ambiguous"]

    hand = hands[0]

    if (len(hand.image_landmarks.landmarks) != 21):
        reasons.append(f"{side}_hand_landmark_count_invalid")

    if hand.association_source == HandAssociationSource.BACKEND_HANDEDNESS:
        if (hand.handedness.strip().lower() != side):
            reasons.append(f"{side}_hand_classification_mismatch")

        if hand.handedness_score is None:
            reasons.append(f"{side}_handedness_score_missing")
        elif (hand.handedness_score < config.min_handedness_score):
            reasons.append(f"{side}_handedness_score_low")

    return reasons


def _landmark_to_dict(landmark: Landmark) -> dict[str, Any]:
    return {
        "index": landmark.index,
        "name": landmark.name,
        "x": landmark.x,
        "y": landmark.y,
        "z": landmark.z,
        "visibility": landmark.visibility,
        "presence": landmark.presence
    }

def _landmark_set_to_dict(landmark_set: LandmarkSet | None) -> dict[str, Any] | None:
    if landmark_set is None:
        return None

    return {
        "coordinate_space": landmark_set.coordinate_space.value,
        "landmarks": [
            _landmark_to_dict(landmark)
            for landmark in landmark_set.landmarks
        ]
    }

def _hand_to_dict(hand: HandObservation) -> dict[str, Any]:
    return {
        "handedness": hand.handedness,
        "handedness_score": hand.handedness_score,
        "association_source": hand.association_source.value,
        "image_landmarks": _landmark_set_to_dict(hand.image_landmarks),
        "world_landmarks": _landmark_set_to_dict(hand.world_landmarks)
    }