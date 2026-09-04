from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import math
from human_tracking.tracking import HumanDerivedKinematics
from human_tracking.vector_math import (
    Vector3,
    _add_vectors,
    _scale_vector,
    _subtract_vectors,
    _vector_norm,
)

class ArmSide(str, Enum):
    LEFT = "left"
    RIGHT = "right"

@dataclass(frozen=True)
class Quaternion:
    x: float
    y: float
    z: float
    w: float

    def __post_init__(self) -> None:
        values = (
            self.x,
            self.y,
            self.z,
            self.w,
        )

        if not all(math.isfinite(value)
            for value in values
        ):
            raise ValueError("Quaternion values must be finite.")

        if self.norm() <= 1e-12:
            raise ValueError("Quaternion norm must be greater than zero.")

    def norm(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w)

    def normalized(self) -> Quaternion:
        norm = self.norm()

        return Quaternion(
            x=self.x / norm,
            y=self.y / norm,
            z=self.z / norm,
            w=self.w / norm,
        )

@dataclass(frozen=True)
class RobotAgnosticPose:
    position: Vector3
    orientation_xyzw: Quaternion

    def __post_init__(self) -> None:
        _validate_vector(self.position, "Robot position")

@dataclass(frozen=True)
class AxisMapping:
    rows: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]

    def __post_init__(self) -> None:
        if len(self.rows) != 3:
            raise ValueError("Axis mapping must contain exactly 3 rows.")

        for row in self.rows:
            if len(row) != 3:
                raise ValueError("Axis mapping rows must contain exactly 3 values.")

            for value in row:
                if value not in (-1, 0, 1, -1.0, 0.0, 1.0):
                    raise ValueError("Axis mapping entries must be -1, 0 or +1.")

        for row in self.rows:
            if sum(
                1
                for value in row
                if value != 0
            ) != 1:
                raise ValueError("Each axis-mapping row must contain exactly one non-zero entry.")

        for column_index in range(3):
            if sum(
                1
                for row in self.rows
                if row[column_index] != 0
            ) != 1:
                raise ValueError("Each axis-mapping column must contain exactly one non-zero entry.")

    @staticmethod
    def identity() -> AxisMapping:
        return AxisMapping(
            rows=(
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        )

    def apply(self, vector: Vector3) -> Vector3:
        components = (vector.x, vector.y, vector.z)
        mapped = []

        for row in self.rows:
            mapped.append(sum(row[index] * components[index]
                    for index in range(3)
                )
            )

        return Vector3(
            x=mapped[0],
            y=mapped[1],
            z=mapped[2],
        )


@dataclass(frozen=True)
class CartesianBounds:
    minimum: Vector3
    maximum: Vector3

    def __post_init__(self) -> None:
        _validate_vector(self.minimum, "Workspace minimum")
        _validate_vector(self.maximum, "Workspace maximum")

        if (
            self.minimum.x > self.maximum.x
            or self.minimum.y > self.maximum.y
            or self.minimum.z > self.maximum.z
        ):
            raise ValueError("Workspace minimum must not exceed maximum.")

    def clamp(self, position: Vector3) -> tuple[Vector3, bool]:
        clamped = Vector3(
            x = min(max(position.x, self.minimum.x), self.maximum.x),
            y = min(max(position.y, self.minimum.y), self.maximum.y),
            z = min(max(position.z, self.minimum.z), self.maximum.z),
        )

        limited = (clamped != position)

        return (clamped, limited)

@dataclass(frozen=True)
class RetargetingConfig:
    axis_mapping: AxisMapping
    scale_robot_per_normalized_body: Vector3
    deadband_normalized_body: Vector3
    workspace_bounds: CartesianBounds
    max_cartesian_speed_robot_per_s: (float | None) = None

    def __post_init__(self) -> None:
        _validate_nonnegative_vector(self.scale_robot_per_normalized_body, "Retargeting scale")
        _validate_nonnegative_vector(self.deadband_normalized_body, "Retargeting deadband")

        if (self.max_cartesian_speed_robot_per_s is not None):
            if not math.isfinite(self.max_cartesian_speed_robot_per_s):
                raise ValueError("Maximum Cartesian speed must be finite.")

            if (self.max_cartesian_speed_robot_per_s <= 0.0):
                raise ValueError("Maximum Cartesian speed must be greater than zero.")

@dataclass(frozen=True)
class RetargetingReference:
    arm_side: ArmSide
    activation_frame_sequence_id: int
    activation_timestamp_s: float
    human_wrist_reference_normalized_body: Vector3
    robot_pose_reference: RobotAgnosticPose

@dataclass(frozen=True)
class RetargetingResult:
    frame_sequence_id: int
    measurement_timestamp_s: float
    active: bool
    valid: bool
    target_pose: RobotAgnosticPose | None
    operator_delta_normalized_body: Vector3 | None
    deadbanded_delta_normalized_body: Vector3 | None
    mapped_delta_robot: Vector3 | None
    workspace_limited: bool
    speed_limited: bool
    reasons: tuple[str, ...]

class RelativeRetargeter:
    def __init__(self, arm_side: ArmSide, config: RetargetingConfig):
        self.arm_side = arm_side
        self.config = config
        self._reference: (RetargetingReference | None) = None
        self._previous_target: (RobotAgnosticPose | None) = None
        self._previous_target_timestamp_s: (float | None) = None
        self._last_input_frame_sequence_id: (int | None) = None
        self._last_input_timestamp_s: (float | None) = None

    @property
    def active(self) -> bool:
        return self._reference is not None

    @property
    def reference(self) -> RetargetingReference | None:
        return self._reference

    def activate(self, human: HumanDerivedKinematics, robot_pose: RobotAgnosticPose) -> RetargetingReference:
        wrist = _selected_wrist(human, self.arm_side)

        if wrist is None:
            raise ValueError("Cannot activate retargeting without valid derived body geometry.")

        if not math.isfinite(human.measurement_timestamp_s):
            raise ValueError("Human measurement timestamp must be finite.")

        normalized_pose = RobotAgnosticPose(
            position = robot_pose.position,
            orientation_xyzw = (robot_pose.orientation_xyzw.normalized()),
        )

        if human.frame_sequence_id < 0:
            raise ValueError("Human frame sequence ID must be non-negative.")

        reference = RetargetingReference(
            arm_side=self.arm_side,
            activation_frame_sequence_id = (human.frame_sequence_id),
            activation_timestamp_s = (human.measurement_timestamp_s),
            human_wrist_reference_normalized_body = (wrist),
            robot_pose_reference = normalized_pose
        )

        self._reference = reference
        self._previous_target = normalized_pose
        self._previous_target_timestamp_s = (human.measurement_timestamp_s)
        self._last_input_frame_sequence_id = (human.frame_sequence_id)
        self._last_input_timestamp_s = (human.measurement_timestamp_s)

        return reference

    def deactivate(self) -> None:
        self._reference = None
        self._previous_target = None
        self._previous_target_timestamp_s = None
        self._last_input_frame_sequence_id = None
        self._last_input_timestamp_s = None

    def update(self, human: HumanDerivedKinematics) -> RetargetingResult:
        timestamp_s = (human.measurement_timestamp_s)

        if not math.isfinite(timestamp_s):
            raise ValueError("Human measurement timestamp must be finite.")

        reference = self._reference

        if reference is None:
            return RetargetingResult(
                frame_sequence_id = (human.frame_sequence_id),
                measurement_timestamp_s = timestamp_s,
                active = False,
                valid = False,
                target_pose = None,
                operator_delta_normalized_body = None,
                deadbanded_delta_normalized_body = None,
                mapped_delta_robot = None,
                workspace_limited = False,
                speed_limited = False,
                reasons = ("retargeting_inactive",),
            )

        if (human.frame_sequence_id <= reference.activation_frame_sequence_id):
            raise ValueError("Retargeting frame sequence IDs must be newer than the activation measurement.")

        if (timestamp_s <= reference.activation_timestamp_s):
            raise ValueError("Retargeting updates must be newer than the activation measurement.")

        if (self._last_input_frame_sequence_id is not None and human.frame_sequence_id <= self._last_input_frame_sequence_id):
            raise ValueError("Retargeting frame sequence IDs must be strictly increasing.")

        if (self._last_input_timestamp_s is not None and timestamp_s <= self._last_input_timestamp_s):
            raise ValueError("Retargeting timestamps must be strictly increasing.")

        self._last_input_frame_sequence_id = human.frame_sequence_id
        self._last_input_timestamp_s = timestamp_s
        current_wrist = _selected_wrist(human, self.arm_side)

        if current_wrist is None:
            return RetargetingResult(
                frame_sequence_id = human.frame_sequence_id,
                measurement_timestamp_s = timestamp_s,
                active = True,
                valid = False,
                target_pose = None,
                operator_delta_normalized_body = None,
                deadbanded_delta_normalized_body = None,
                mapped_delta_robot = None,
                workspace_limited = False,
                speed_limited = False,
                reasons = ("body_derived_kinematics_unavailable",),
            )

        raw_delta = _subtract_vectors(current_wrist, reference.human_wrist_reference_normalized_body)
        deadbanded_delta = Vector3(
            x = _apply_deadband(raw_delta.x, self.config.deadband_normalized_body.x),
            y = _apply_deadband(raw_delta.y, self.config.deadband_normalized_body.y),
            z = _apply_deadband(raw_delta.z, self.config.deadband_normalized_body.z)
        )

        scaled_body_delta = Vector3(
            x = (deadbanded_delta.x * self.config.scale_robot_per_normalized_body.x),
            y = (deadbanded_delta.y * self.config.scale_robot_per_normalized_body.y),
            z = (deadbanded_delta.z * self.config.scale_robot_per_normalized_body.z)
        )

        mapped_delta = self.config.axis_mapping.apply(scaled_body_delta)
        unconstrained_position = _add_vectors(reference.robot_pose_reference.position, mapped_delta)
        (workspace_position, workspace_limited) = self.config.workspace_bounds.clamp(unconstrained_position)
        (final_position, speed_limited) = self._apply_speed_limit(desired_position=workspace_position, timestamp_s = timestamp_s)

        target_pose = RobotAgnosticPose(position = final_position, orientation_xyzw = (reference.robot_pose_reference.orientation_xyzw))
        self._previous_target = target_pose
        self._previous_target_timestamp_s = timestamp_s

        return RetargetingResult(
            frame_sequence_id = (human.frame_sequence_id),
            measurement_timestamp_s = timestamp_s,
            active = True,
            valid = True,
            target_pose = target_pose,
            operator_delta_normalized_body = raw_delta,
            deadbanded_delta_normalized_body = deadbanded_delta,
            mapped_delta_robot = mapped_delta,
            workspace_limited = workspace_limited,
            speed_limited = speed_limited,
            reasons = ()
        )

    def _apply_speed_limit(self, desired_position: Vector3, timestamp_s: float) -> tuple[Vector3, bool]:
        max_speed = self.config.max_cartesian_speed_robot_per_s

        if max_speed is None:
            return (desired_position, False)

        previous_target = self._previous_target
        previous_timestamp_s = self._previous_target_timestamp_s

        if previous_target is None or previous_timestamp_s is None:
            return (desired_position, False)

        dt_s = timestamp_s - previous_timestamp_s

        if dt_s <= 0.0:
            raise ValueError("Retargeting timestamps must be strictly increasing.")

        allowed_step = max_speed * dt_s
        step = _subtract_vectors(desired_position, previous_target.position)
        step_norm = _vector_norm(step)

        if step_norm <= allowed_step or step_norm <= 1e-12:
            return (desired_position, False)

        limited_step = _scale_vector(step, allowed_step / step_norm)
        limited_position = _add_vectors(previous_target.position, limited_step)

        return (limited_position,True)

def _selected_wrist(human: HumanDerivedKinematics, arm_side: ArmSide) -> Vector3 | None:
    if human.body is None:
        return None

    if arm_side == ArmSide.LEFT:
        return human.body.left_wrist_displacement_normalized_body

    if arm_side == ArmSide.RIGHT:
        return human.body.right_wrist_displacement_normalized_body

    raise ValueError (f"Unsupported arm side: {arm_side}")

def _apply_deadband(value: float, threshold: float) -> float:
    if abs(value) <= threshold:
        return 0.0

    return value

def _validate_vector(vector: Vector3, label: str) -> None:
    values = (
        vector.x,
        vector.y,
        vector.z,
    )

    if not all(
        math.isfinite(value)
        for value in values
    ):
        raise ValueError(f"{label} values must be finite.")

def _validate_nonnegative_vector(vector: Vector3, label: str) -> None:
    _validate_vector(vector, label)

    if (vector.x < 0.0 or vector.y < 0.0 or vector.z < 0.0):
        raise ValueError(f"{label} values must be non-negative.")