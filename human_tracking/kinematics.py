from __future__ import annotations
from dataclasses import dataclass
import math
from human_tracking.observation import ControlArm, Landmark, LandmarkSet
from human_tracking.smoothing import SmoothedHumanLandmarks
from human_tracking.vector_math import (
    Vector3,
    _add_vectors,
    _scale_vector,
    _subtract_vectors,
    _vector_norm
)

@dataclass(frozen=True)
class DerivedGeometryConfig:
    selected_control_arm: ControlArm
    min_length_model_world: float = 1e-5
    min_palm_width_to_palm_length_ratio: float | None = None

    def __post_init__(self) -> None:
        try:
            selected_control_arm = ControlArm(self.selected_control_arm)

        except ValueError as exc:
            raise ValueError("selected_control_arm must be 'left' or 'right'.") from exc

        object.__setattr__(self, "selected_control_arm", selected_control_arm)

        if not math.isfinite(self.min_length_model_world):
            raise ValueError("min_length_model_world must be finite.")

        if self.min_length_model_world <= 0.0:
            raise ValueError("min_length_model_world must be greater than zero.")

        ratio = self.min_palm_width_to_palm_length_ratio
        if ratio is not None:
            if not math.isfinite(ratio):
                raise ValueError("min_palm_width_to_palm_length_ratio must be finite.")
            if ratio <= 0.0:
                raise ValueError("min_palm_width_to_palm_length_ratio must be greater than zero when configured.")

@dataclass(frozen=True)
class BodyRelativeFrame:
    origin_model_world: Vector3
    right_axis_model_world: Vector3
    up_axis_model_world: Vector3
    normal_axis_model_world: Vector3

@dataclass(frozen=True)
class BodyDerivedKinematics:
    frame: BodyRelativeFrame
    shoulder_width_model_world: float
    torso_height_model_world: float
    left_arm_length_model_world: float | None
    right_arm_length_model_world: float | None
    left_upper_arm_direction_body: Vector3 | None
    left_forearm_direction_body: Vector3 | None
    right_upper_arm_direction_body: Vector3 | None
    right_forearm_direction_body: Vector3 | None
    left_wrist_displacement_normalized_body: Vector3 | None
    right_wrist_displacement_normalized_body: Vector3 | None

@dataclass(frozen=True)
class HandDerivedKinematics:
    palm_width_model_world: float
    palm_length_model_world: float
    pinch_ratio: float
    index_direction_model_world: Vector3
    palm_normal_model_world: Vector3

@dataclass(frozen=True)
class HumanDerivedKinematics:
    frame_sequence_id: int
    measurement_timestamp_s: float
    body: BodyDerivedKinematics | None
    left_hand: HandDerivedKinematics | None
    right_hand: HandDerivedKinematics | None
    body_reasons: tuple[str, ...]
    left_hand_reasons: tuple[str, ...]
    right_hand_reasons: tuple[str, ...]

def derive_human_kinematics(smoothed: SmoothedHumanLandmarks, config: DerivedGeometryConfig) -> HumanDerivedKinematics:
    body, body_reasons = _derive_body_kinematics(smoothed.body.world_landmarks, config)
    left_hand, left_reasons = _derive_hand_kinematics(smoothed.left_hand.world_landmarks, config, side_name="left")
    right_hand, right_reasons = _derive_hand_kinematics(smoothed.right_hand.world_landmarks, config, side_name="right")

    return HumanDerivedKinematics(
        frame_sequence_id = smoothed.frame_sequence_id,
        measurement_timestamp_s = smoothed.measurement_timestamp_s,
        body = body,
        left_hand = left_hand,
        right_hand = right_hand,
        body_reasons = tuple(body_reasons),
        left_hand_reasons = tuple(left_reasons),
        right_hand_reasons = tuple(right_reasons)
    )

def _derive_body_kinematics(landmark_set: LandmarkSet | None, config: DerivedGeometryConfig) -> tuple[BodyDerivedKinematics | None, list[str]]:
    if landmark_set is None:
        return (None, ["body_world_landmarks_unavailable"])
    if len(landmark_set.landmarks) != 33:
        return (None, ["body_world_landmark_count_invalid"])

    landmarks = landmark_set.landmarks

    left_shoulder = _landmark_vector(landmarks[11])
    right_shoulder = _landmark_vector(landmarks[12])
    left_hip = _landmark_vector(landmarks[23])
    right_hip = _landmark_vector(landmarks[24])
    shoulder_midpoint = _scale_vector(_add_vectors(left_shoulder, right_shoulder), 0.5)
    hip_midpoint = _scale_vector(_add_vectors(left_hip, right_hip), 0.5)
    shoulder_vector = _subtract_vectors(right_shoulder, left_shoulder)
    torso_vector = _subtract_vectors(shoulder_midpoint, hip_midpoint)
    shoulder_width = _vector_norm(shoulder_vector)
    torso_height = _vector_norm(torso_vector)

    if (shoulder_width <= config.min_length_model_world):
        return (None, ["body_shoulder_width_degenerate"])
    if (torso_height <= config.min_length_model_world):
        return (None, ["body_torso_height_degenerate"])

    right_axis = _normalize_vector(shoulder_vector, config.min_length_model_world)
    torso_up_seed = _normalize_vector(torso_vector, config.min_length_model_world)
    normal_seed = _cross_vectors(right_axis, torso_up_seed)

    if (_vector_norm(normal_seed) <= config.min_length_model_world):
        return (None, ["body_frame_degenerate"])

    normal_axis = _normalize_vector(normal_seed, config.min_length_model_world)
    up_axis = _normalize_vector(_cross_vectors(normal_axis, right_axis), config.min_length_model_world)

    if config.selected_control_arm == ControlArm.LEFT:
        side_name = "left"
        shoulder = left_shoulder
        elbow = _landmark_vector(landmarks[13])
        wrist = _landmark_vector(landmarks[15])

    elif config.selected_control_arm == ControlArm.RIGHT:
        side_name = "right"
        shoulder = right_shoulder
        elbow = _landmark_vector(landmarks[14])
        wrist = _landmark_vector(landmarks[16])

    else:
        raise ValueError(f"Unsupported control arm: {config.selected_control_arm}")

    upper_world = _subtract_vectors(elbow, shoulder)
    forearm_world = _subtract_vectors(wrist, elbow)
    upper_length = _vector_norm(upper_world)
    forearm_length = _vector_norm(forearm_world)

    if (upper_length <= config.min_length_model_world):
        return (None, [f"{side_name}_upper_arm_length_degenerate"])
    if (forearm_length <= config.min_length_model_world):
        return (None, [f"{side_name}_forearm_length_degenerate"])

    arm_length = upper_length + forearm_length
    upper_direction_body = _vector_to_body_frame(_normalize_vector(upper_world, config.min_length_model_world), right_axis, up_axis, normal_axis,)
    forearm_direction_body = _vector_to_body_frame(_normalize_vector(forearm_world, config.min_length_model_world), right_axis, up_axis, normal_axis)
    wrist_from_shoulder = _subtract_vectors(wrist, shoulder)
    wrist_normalized_world = _scale_vector(wrist_from_shoulder, 1.0 / arm_length)
    wrist_displacement_normalized_body = _vector_to_body_frame(wrist_normalized_world, right_axis, up_axis, normal_axis)

    frame = BodyRelativeFrame(
        origin_model_world = hip_midpoint,
        right_axis_model_world = right_axis,
        up_axis_model_world = up_axis,
        normal_axis_model_world = normal_axis
    )

    if config.selected_control_arm == ControlArm.LEFT:
        left_arm_length = arm_length
        right_arm_length = None
        left_upper_direction = upper_direction_body
        left_forearm_direction = forearm_direction_body
        right_upper_direction = None
        right_forearm_direction = None
        left_wrist_displacement = wrist_displacement_normalized_body
        right_wrist_displacement = None

    else:
        left_arm_length = None
        right_arm_length = arm_length
        left_upper_direction = None
        left_forearm_direction = None
        right_upper_direction = upper_direction_body
        right_forearm_direction = forearm_direction_body
        left_wrist_displacement = None
        right_wrist_displacement = wrist_displacement_normalized_body

    return (
        BodyDerivedKinematics(
            frame = frame,
            shoulder_width_model_world = shoulder_width,
            torso_height_model_world = torso_height,
            left_arm_length_model_world = left_arm_length,
            right_arm_length_model_world = right_arm_length,
            left_upper_arm_direction_body = left_upper_direction,
            left_forearm_direction_body = left_forearm_direction,
            right_upper_arm_direction_body = right_upper_direction,
            right_forearm_direction_body = right_forearm_direction,
            left_wrist_displacement_normalized_body = left_wrist_displacement,
            right_wrist_displacement_normalized_body = right_wrist_displacement
            ),[])

def _derive_hand_kinematics(landmark_set: LandmarkSet | None, config: DerivedGeometryConfig, side_name: str) -> tuple[HandDerivedKinematics | None, list[str]]:
    if landmark_set is None:
        return (None, [f"{side_name}_hand_world_landmarks_unavailable"])
    if len(landmark_set.landmarks) != 21:
        return (None,[f"{side_name}_hand_world_landmark_count_invalid"])

    landmarks = landmark_set.landmarks
    wrist = _landmark_vector(landmarks[0])
    thumb_tip = _landmark_vector(landmarks[4])
    index_mcp = _landmark_vector(landmarks[5])
    index_tip = _landmark_vector(landmarks[8])
    middle_mcp = _landmark_vector(landmarks[9])
    pinky_mcp = _landmark_vector(landmarks[17])
    palm_width_vector = _subtract_vectors(index_mcp, pinky_mcp)
    palm_width = _vector_norm(palm_width_vector)

    if (palm_width <= config.min_length_model_world):
        return (None, [f"{side_name}_hand_palm_width_degenerate"])

    palm_length = _vector_norm(_subtract_vectors(middle_mcp, wrist))

    if palm_length <= config.min_length_model_world:
        return (None, [f"{side_name}_hand_palm_length_degenerate"])

    if config.min_palm_width_to_palm_length_ratio is not None:
        palm_width_to_length_ratio = palm_width / palm_length

        if (palm_width_to_length_ratio < config.min_palm_width_to_palm_length_ratio):
            return (None, [f"{side_name}_hand_palm_shape_implausible"])

    pinch_distance = _vector_norm(_subtract_vectors(thumb_tip, index_tip))
    index_vector = _subtract_vectors(index_tip, wrist)

    if (_vector_norm(index_vector) <= config.min_length_model_world):
        return (None, [f"{side_name}_hand_index_direction_degenerate"])

    wrist_to_index_mcp = _subtract_vectors(index_mcp, wrist)
    wrist_to_pinky_mcp = _subtract_vectors(pinky_mcp, wrist)
    palm_normal_seed = _cross_vectors(wrist_to_index_mcp, wrist_to_pinky_mcp)

    if _vector_norm(palm_normal_seed) <= config.min_length_model_world:
        return (None, [f"{side_name}_hand_palm_normal_degenerate"])

    return (
        HandDerivedKinematics(
            palm_width_model_world = palm_width,
            palm_length_model_world = palm_length,
            pinch_ratio = pinch_distance / palm_length,
            index_direction_model_world = _normalize_vector(index_vector, config.min_length_model_world),
            palm_normal_model_world = _normalize_vector(palm_normal_seed, config.min_length_model_world)
        ),[])

def _landmark_vector(landmark: Landmark) -> Vector3:
    values = (
        float(landmark.x),
        float(landmark.y),
        float(landmark.z)
    )

    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Landmark {landmark.name} contains non-finite coordinates.")

    return Vector3(
        x = values[0],
        y = values[1],
        z = values[2]
    )

def _normalize_vector(vector: Vector3, min_norm: float) -> Vector3:
    norm = _vector_norm(vector)
    if norm <= min_norm:
        raise ValueError("Cannot normalize a degenerate vector.")
    return _scale_vector(vector, 1.0 / norm)

def _dot_vectors(first: Vector3, second: Vector3) -> float:
    return (first.x * second.x + first.y * second.y + first.z * second.z)

def _cross_vectors(first: Vector3, second: Vector3) -> Vector3:
    return Vector3(
        x = first.y * second.z - first.z * second.y,
        y = first.z * second.x - first.x * second.z,
        z = first.x * second.y - first.y * second.x
    )

def _vector_to_body_frame(vector_model_world: Vector3, right_axis: Vector3, up_axis: Vector3, normal_axis: Vector3) -> Vector3:
    return Vector3(
        x = _dot_vectors(vector_model_world, right_axis),
        y = _dot_vectors(vector_model_world, up_axis),
        z = _dot_vectors(vector_model_world, normal_axis)
    )