from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
import platform
from pathlib import Path
import sys
import time

import cv2
import mediapipe as mp

from human_tracking.acquisition import CameraAcquisition
from human_tracking.mediapipe_adapter import (
    MediaPipeObservationAdapter,
)
from human_tracking.mediapipe_backend import (
    MediaPipeTrackingBackend,
)
from human_tracking.observation import (
    ObservationValidationConfig,
    evaluate_observation,
)
from human_tracking.tracking import (
    DerivedGeometryConfig,
    HumanLandmarkSmoother,
    HumanTemporalTracker,
    LandmarkSmoothingConfig,
    TemporalTrackingConfig,
    TrackingState,
    Vector3,
    derive_human_kinematics,
)
from human_tracking.retargeting import (
    ArmSide,
    AxisMapping,
    CartesianBounds,
    Quaternion,
    RelativeRetargeter,
    RetargetingConfig,
    RobotAgnosticPose,
)


WINDOW_NAME = "H2/H4/H5/H6/H8 - Human Tracking and Virtual Retargeting"

# OpenCV uses BGR color ordering.
# Equivalent RGB color: #39FF14.
TEXT_COLOR = (20, 255, 57)
TEXT_OUTLINE_COLOR = (0, 0, 0)
SMOOTHED_COLOR = (255, 0, 255)


POSE_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
)


HAND_CONNECTIONS = (
    (0, 1),
    (1, 5),
    (5, 9),
    (9, 13),
    (13, 17),
    (17, 0),
    (1, 2),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (7, 8),
    (9, 10),
    (10, 11),
    (11, 12),
    (13, 14),
    (14, 15),
    (15, 16),
    (17, 18),
    (18, 19),
    (19, 20),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Live MediaPipe Pose + Hand tracking "
            "with H4 per-frame and temporal validity gates."
        )
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera index. Default: 0",
    )

    parser.add_argument(
        "--pose-model",
        type=Path,
        default=Path(
            "models/pose_landmarker_full.task"
        ),
    )

    parser.add_argument(
        "--hand-model",
        type=Path,
        default=Path(
            "models/hand_landmarker.task"
        ),
    )

    parser.add_argument(
        "--smoothing-time-constant",
        type=float,
        default=0.10,
        help=(
            "Low-pass smoothing time constant in seconds. "
            "Default: 0.10"
        ),
    )

    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help=(
            "Directory for H7/H8 JSONL diagnostic logs. "
            "Default: logs"
        ),
    )

    parser.add_argument(
        "--control-arm",
        choices=("left", "right"),
        default="right",
        help=(
            "Arm used by the H8 virtual retargeting check. "
            "Default: right"
        ),
    )

    return parser.parse_args()


def landmark_to_pixel(
    landmark,
    width: int,
    height: int,
):
    x = float(landmark.x)
    y = float(landmark.y)

    if not (
        math.isfinite(x)
        and math.isfinite(y)
    ):
        return None

    if x < 0.0 or x > 1.0:
        return None

    if y < 0.0 or y > 1.0:
        return None

    px = int(
        round(x * (width - 1))
    )

    py = int(
        round(y * (height - 1))
    )

    return px, py


def draw_landmark_set(
    image,
    landmarks,
    connections,
    point_color,
    line_color,
):
    height, width = image.shape[:2]

    points = [
        landmark_to_pixel(
            landmark,
            width,
            height,
        )
        for landmark in landmarks
    ]

    for start, end in connections:
        if start >= len(points):
            continue

        if end >= len(points):
            continue

        start_point = points[start]
        end_point = points[end]

        if (
            start_point is None
            or end_point is None
        ):
            continue

        cv2.line(
            image,
            start_point,
            end_point,
            line_color,
            2,
            cv2.LINE_AA,
        )

    for point in points:
        if point is None:
            continue

        cv2.circle(
            image,
            point,
            3,
            point_color,
            -1,
            cv2.LINE_AA,
        )

    return points


def draw_text(
    image,
    text: str,
    position: tuple[int, int],
    font_scale: float = 0.50,
    thickness: int = 1,
):
    """
    Draw neon-green diagnostic text with a black outline.

    The outline keeps the text readable against both bright and dark
    parts of the camera image.
    """

    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        TEXT_OUTLINE_COLOR,
        thickness + 3,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        TEXT_COLOR,
        thickness,
        cv2.LINE_AA,
    )


def mean_xy_separation_px(
    raw_landmarks,
    smoothed_landmarks,
    width: int,
    height: int,
) -> float | None:
    if len(raw_landmarks) != len(smoothed_landmarks):
        return None

    distances = []

    for raw, smoothed in zip(
        raw_landmarks,
        smoothed_landmarks,
    ):
        raw_x = float(raw.x)
        raw_y = float(raw.y)
        smooth_x = float(smoothed.x)
        smooth_y = float(smoothed.y)

        if not all(
            math.isfinite(value)
            for value in (
                raw_x,
                raw_y,
                smooth_x,
                smooth_y,
            )
        ):
            continue

        dx_px = (raw_x - smooth_x) * width
        dy_px = (raw_y - smooth_y) * height

        distances.append(
            math.hypot(dx_px, dy_px)
        )

    if not distances:
        return None

    return sum(distances) / len(distances)


def extract_handedness(
    hand_result,
    hand_index: int,
):
    if hand_index >= len(
        hand_result.handedness
    ):
        return "Unknown", None

    categories = (
        hand_result.handedness[
            hand_index
        ]
    )

    if not categories:
        return "Unknown", None

    category = categories[0]

    name = getattr(
        category,
        "category_name",
        None,
    )

    if not name:
        name = "Unknown"

    score = getattr(
        category,
        "score",
        None,
    )

    if score is not None:
        score = float(score)

    return str(name), score


def first_reason(
    reasons: tuple[str, ...],
) -> str:
    if not reasons:
        return "none"

    return reasons[0]


def percentage(
    count: int,
    total: int,
) -> float:
    if total <= 0:
        return 0.0

    return 100.0 * count / total


def format_state_counts(
    counts: dict[TrackingState, int],
) -> str:
    return ", ".join(
        f"{state.value}={counts[state]}"
        for state in TrackingState
    )



LOG_SCHEMA_NAME = "human_tracking_diagnostic"
LOG_SCHEMA_VERSION = 3


def json_safe(value):
    """
    Convert supported diagnostic values into JSON-safe structures.
    """

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, tuple):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, list):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if hasattr(value, "__dict__"):
        return {
            key: json_safe(item)
            for key, item in vars(value).items()
        }

    raise TypeError(
        f"Unsupported JSON diagnostic value: "
        f"{type(value).__name__}"
    )


def tracking_channel_to_dict(channel):
    return {
        "state": channel.state.value,
        "current_frame_valid": (
            channel.current_frame_valid
        ),
        "valid_for_control": (
            channel.valid_for_control
        ),
        "consecutive_valid_frames": (
            channel.consecutive_valid_frames
        ),
        "age_since_last_valid_s": (
            channel.age_since_last_valid_s
        ),
        "reasons": list(
            channel.reasons
        ),
    }


def vector_to_dict(vector):
    if vector is None:
        return None

    return {
        "x": float(vector.x),
        "y": float(vector.y),
        "z": float(vector.z),
    }


def derived_to_dict(derived):
    result = {
        "frame_sequence_id": (
            derived.frame_sequence_id
        ),
        "measurement_timestamp_s": (
            derived.measurement_timestamp_s
        ),
        "body": None,
        "left_hand": None,
        "right_hand": None,
        "body_reasons": list(
            derived.body_reasons
        ),
        "left_hand_reasons": list(
            derived.left_hand_reasons
        ),
        "right_hand_reasons": list(
            derived.right_hand_reasons
        ),
    }

    if derived.body is not None:
        body = derived.body

        result["body"] = {
            "frame": {
                "origin_model_world": (
                    vector_to_dict(
                        body.frame.origin_model_world
                    )
                ),
                "right_axis_model_world": (
                    vector_to_dict(
                        body.frame.right_axis_model_world
                    )
                ),
                "up_axis_model_world": (
                    vector_to_dict(
                        body.frame.up_axis_model_world
                    )
                ),
                "normal_axis_model_world": (
                    vector_to_dict(
                        body.frame.normal_axis_model_world
                    )
                ),
            },
            "shoulder_width_model_world": (
                body.shoulder_width_model_world
            ),
            "torso_height_model_world": (
                body.torso_height_model_world
            ),
            "left_arm_length_model_world": (
                body.left_arm_length_model_world
            ),
            "right_arm_length_model_world": (
                body.right_arm_length_model_world
            ),
            "left_upper_arm_direction_body": (
                vector_to_dict(
                    body.left_upper_arm_direction_body
                )
            ),
            "left_forearm_direction_body": (
                vector_to_dict(
                    body.left_forearm_direction_body
                )
            ),
            "right_upper_arm_direction_body": (
                vector_to_dict(
                    body.right_upper_arm_direction_body
                )
            ),
            "right_forearm_direction_body": (
                vector_to_dict(
                    body.right_forearm_direction_body
                )
            ),
            "left_wrist_displacement_normalized_body": (
                vector_to_dict(
                    body
                    .left_wrist_displacement_normalized_body
                )
            ),
            "right_wrist_displacement_normalized_body": (
                vector_to_dict(
                    body
                    .right_wrist_displacement_normalized_body
                )
            ),
        }

    if derived.left_hand is not None:
        left = derived.left_hand

        result["left_hand"] = {
            "palm_width_model_world": (
                left.palm_width_model_world
            ),
            "pinch_ratio": (
                left.pinch_ratio
            ),
            "index_direction_model_world": (
                vector_to_dict(
                    left.index_direction_model_world
                )
            ),
            "palm_normal_model_world": (
                vector_to_dict(
                    left.palm_normal_model_world
                )
            ),
        }

    if derived.right_hand is not None:
        right = derived.right_hand

        result["right_hand"] = {
            "palm_width_model_world": (
                right.palm_width_model_world
            ),
            "pinch_ratio": (
                right.pinch_ratio
            ),
            "index_direction_model_world": (
                vector_to_dict(
                    right.index_direction_model_world
                )
            ),
            "palm_normal_model_world": (
                vector_to_dict(
                    right.palm_normal_model_world
                )
            ),
        }

    return result


def file_identity(
    path: Path,
) -> dict:
    """
    Record enough information to identify a model asset exactly.
    """

    resolved = (
        path.expanduser().resolve()
    )

    if not resolved.is_file():
        raise FileNotFoundError(
            f"Model asset not found: {resolved}"
        )

    digest = hashlib.sha256()

    with resolved.open("rb") as file:
        while True:
            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return {
        "path": str(resolved),
        "size_bytes": (
            resolved.stat().st_size
        ),
        "sha256": digest.hexdigest(),
    }



def retargeting_result_to_dict(result):
    if result is None:
        return None

    return {
        "frame_sequence_id": (
            result.frame_sequence_id
        ),
        "measurement_timestamp_s": (
            result.measurement_timestamp_s
        ),
        "active": result.active,
        "valid": result.valid,
        "target_pose": (
            None
            if result.target_pose is None
            else {
                "position": vector_to_dict(
                    result.target_pose.position
                ),
                "orientation_xyzw": {
                    "x": (
                        result
                        .target_pose
                        .orientation_xyzw
                        .x
                    ),
                    "y": (
                        result
                        .target_pose
                        .orientation_xyzw
                        .y
                    ),
                    "z": (
                        result
                        .target_pose
                        .orientation_xyzw
                        .z
                    ),
                    "w": (
                        result
                        .target_pose
                        .orientation_xyzw
                        .w
                    ),
                },
            }
        ),
        "operator_delta_normalized_body": (
            vector_to_dict(
                result
                .operator_delta_normalized_body
            )
        ),
        "deadbanded_delta_normalized_body": (
            vector_to_dict(
                result
                .deadbanded_delta_normalized_body
            )
        ),
        "mapped_delta_robot": (
            vector_to_dict(
                result.mapped_delta_robot
            )
        ),
        "workspace_limited": (
            result.workspace_limited
        ),
        "speed_limited": (
            result.speed_limited
        ),
        "reasons": list(
            result.reasons
        ),
    }

class JsonlDiagnosticLogger:
    """
    H7 append-only JSON Lines logger.

    One complete JSON object is written per line and flushed immediately,
    mirroring the durable per-cycle logging pattern used in Part 1.
    """

    def __init__(
        self,
        log_dir: Path,
    ):
        self.log_dir = (
            log_dir.expanduser().resolve()
        )

        self.log_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        timestamp_utc = (
            datetime.now(timezone.utc)
            .strftime("%Y%m%dT%H%M%S_%fZ")
        )

        self.path = (
            self.log_dir
            / (
                "human_tracking_"
                f"{timestamp_utc}.jsonl"
            )
        )

        self._file = self.path.open(
            "x",
            encoding="utf-8",
            newline="\n",
        )

        self.record_count = 0

    def write(
        self,
        record_type: str,
        payload: dict,
    ) -> None:
        record = {
            "schema_name": LOG_SCHEMA_NAME,
            "schema_version": LOG_SCHEMA_VERSION,
            "record_type": record_type,
            **payload,
        }

        self._file.write(
            json.dumps(
                json_safe(record),
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
        )

        self._file.flush()
        self.record_count += 1

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

def main():
    args = parse_args()

    print("=" * 60)
    print("H2/H4/H5/H6/H7/H8 - Live Human Tracking + Virtual Retargeting Check (schema v3)")
    print("=" * 60)

    print(f"Camera index: {args.camera}")
    print(f"Pose model:   {args.pose_model}")
    print(f"Hand model:   {args.hand_model}")
    print()

    validation_config = (
        ObservationValidationConfig(
            min_body_visibility=0.50,
            min_body_presence=0.50,
            min_handedness_score=0.50,
        )
    )

    temporal_config = TemporalTrackingConfig(
        consecutive_valid_required=3,
        dropout_timeout_s=0.25,
    )

    smoothing_config = LandmarkSmoothingConfig(
        time_constant_s=args.smoothing_time_constant,
    )

    derived_config = DerivedGeometryConfig(
        min_length_model_world=1e-5,
    )

    control_arm = ArmSide(
        args.control_arm
    )

    h8_config = RetargetingConfig(
        axis_mapping=AxisMapping.identity(),
        scale_robot_per_normalized_body=Vector3(
            x=1.0,
            y=1.0,
            z=1.0,
        ),
        deadband_normalized_body=Vector3(
            x=0.0,
            y=0.0,
            z=0.0,
        ),
        workspace_bounds=CartesianBounds(
            minimum=Vector3(
                x=-100.0,
                y=-100.0,
                z=-100.0,
            ),
            maximum=Vector3(
                x=100.0,
                y=100.0,
                z=100.0,
            ),
        ),
        max_cartesian_speed_robot_per_s=None,
    )

    h8_virtual_reference_pose = RobotAgnosticPose(
        position=Vector3(
            x=0.0,
            y=0.0,
            z=0.0,
        ),
        orientation_xyzw=Quaternion(
            x=0.0,
            y=0.0,
            z=0.0,
            w=1.0,
        ),
    )

    retargeter = RelativeRetargeter(
        arm_side=control_arm,
        config=h8_config,
    )

    print("Current diagnostic thresholds:")

    print(
        "  Minimum body visibility: "
        f"{validation_config.min_body_visibility:.2f}"
    )

    print(
        "  Minimum body presence:   "
        f"{validation_config.min_body_presence:.2f}"
    )

    print(
        "  Minimum handedness score: "
        f"{validation_config.min_handedness_score:.2f}"
    )

    print()

    print("Temporal diagnostic settings:")

    print(
        "  Consecutive valid frames required: "
        f"{temporal_config.consecutive_valid_required}"
    )

    print(
        "  Dropout timeout: "
        f"{temporal_config.dropout_timeout_s:.2f} s"
    )

    print()
    print("H5 smoothing settings:")

    print(
        "  Time constant: "
        f"{smoothing_config.time_constant_s:.3f} s"
    )

    print()
    print("H6 derived-geometry settings:")

    print(
        "  Minimum non-degenerate model-world length: "
        f"{derived_config.min_length_model_world:.1e}"
    )

    print()
    print("H8 virtual-retargeting settings:")

    print(
        "  Control arm: "
        f"{control_arm.value}"
    )

    print(
        "  Axis mapping: identity"
    )

    print(
        "  Scale: 1 virtual target unit per "
        "normalized body unit"
    )

    print(
        "  Deadband: 0 for this diagnostic"
    )

    print(
        "  Physical workspace/speed limits: NOT selected"
    )

    print()

    print(
        "These are preliminary diagnostic thresholds, "
        "not final thesis values."
    )

    print(
        "Left/Right is still the raw MediaPipe "
        "handedness classification."
    )

    print()

    try:
        acquisition = CameraAcquisition(
            camera_index=args.camera
        )

        acquisition.open()

    except Exception as exc:
        print("RESULT: FAIL")

        print(
            f"Camera error: "
            f"{type(exc).__name__}: {exc}"
        )

        raise SystemExit(1)

    try:
        backend = MediaPipeTrackingBackend(
            pose_model_path=args.pose_model,
            hand_model_path=args.hand_model,
        )

    except Exception as exc:
        acquisition.close()

        print("RESULT: FAIL")

        print(
            f"Backend error: "
            f"{type(exc).__name__}: {exc}"
        )

        raise SystemExit(1)

    adapter = MediaPipeObservationAdapter()

    tracker = HumanTemporalTracker(
        temporal_config
    )

    smoother = HumanLandmarkSmoother(
        smoothing_config
    )

    measurement_started = False
    measurement_start_s = None

    logger = None
    log_path = None

    retargeting_result = None

    retargeting_activation_count = 0
    retargeting_active_frame_count = 0
    retargeting_valid_target_count = 0
    retargeting_invalid_target_count = 0
    retargeting_workspace_limited_count = 0
    retargeting_speed_limited_count = 0

    max_operator_delta_norm = 0.0
    max_virtual_target_norm = 0.0

    frame_count = 0

    pose_frame_count = 0
    hand_frame_count = 0

    total_hand_detections = 0
    max_hands_seen = 0
    two_hand_raw_frame_count = 0
    left_raw_detected_count = 0
    right_raw_detected_count = 0

    visualization_frame_count = 0

    body_valid_count = 0
    left_hand_valid_count = 0
    right_hand_valid_count = 0
    both_hands_valid_count = 0

    body_temporal_valid_count = 0
    left_temporal_valid_count = 0
    right_temporal_valid_count = 0
    both_hands_temporal_valid_count = 0

    body_rejection_reasons = Counter()
    left_hand_rejection_reasons = Counter()
    right_hand_rejection_reasons = Counter()

    smoothed_body_frame_count = 0
    smoothed_left_frame_count = 0
    smoothed_right_frame_count = 0

    body_separation_sum_px = 0.0
    body_separation_samples = 0
    left_separation_sum_px = 0.0
    left_separation_samples = 0
    right_separation_sum_px = 0.0
    right_separation_samples = 0

    derived_body_frame_count = 0
    derived_left_frame_count = 0
    derived_right_frame_count = 0

    body_derived_reasons = Counter()
    left_derived_reasons = Counter()
    right_derived_reasons = Counter()

    left_pinch_ratio_sum = 0.0
    left_pinch_ratio_samples = 0
    left_pinch_ratio_min = None
    left_pinch_ratio_max = None

    right_pinch_ratio_sum = 0.0
    right_pinch_ratio_samples = 0
    right_pinch_ratio_min = None
    right_pinch_ratio_max = None

    left_wrist_norm_sum = 0.0
    left_wrist_norm_samples = 0
    right_wrist_norm_sum = 0.0
    right_wrist_norm_samples = 0

    body_state_counts = {
        state: 0
        for state in TrackingState
    }

    left_state_counts = {
        state: 0
        for state in TrackingState
    }

    right_state_counts = {
        state: 0
        for state in TrackingState
    }

    total_processing_s = 0.0

    try:
        print("Models loaded successfully.")
        print()

        print(
            "Preview starts immediately, but statistics are "
            "NOT collected yet."
        )

        print(
            "Stand up, move into the intended test position, "
            "and make sure the required body landmarks are visible."
        )

        print(
            "When you are ready, press S in the camera window "
            "to start the measurement."
        )

        print(
            "After S, hold still briefly, then move your arms "
            "slowly through several comfortable positions."
        )

        print(
            "Open and close a thumb-index pinch several times "
            "with each hand while keeping the hands visible."
        )

        print(
            "Raw pose is green, raw hands are yellow, and "
            "smoothed control-valid landmarks are magenta."
        )

        print(
            "H6 reports body-relative wrist displacement and "
            "normalized pinch ratios; it does not issue gestures "
            "or robot commands."
        )

        print(
            "H7 creates one JSONL log when S is pressed. "
            "Preview frames before S are not logged."
        )

        print(
            "After S, adopt a comfortable neutral pose and press R "
            "to capture the H8 relative-control reference."
        )

        print(
            "Then move the selected wrist slowly in several directions. "
            "Press R again at any time to recapture the reference."
        )

        print(
            "H8 uses virtual unitless target coordinates only. "
            "No robot, physical workspace, or physical speed limit "
            "is being assumed."
        )

        print(
            "Press Q to stop the test."
        )

        print("-" * 60)

        while True:
            frame = acquisition.read()

            output = backend.process(
                frame
            )

            observation = adapter.convert(
                frame,
                output,
            )

            validity = evaluate_observation(
                observation,
                validation_config,
            )

            tracking = tracker.update(
                observation,
                validity,
            )

            smoothed = smoother.update(
                observation,
                tracking,
            )

            derived = derive_human_kinematics(
                smoothed,
                derived_config,
            )

            if measurement_started:
                retargeting_result = (
                    retargeter.update(
                        derived
                    )
                )
            else:
                retargeting_result = None

            display = (
                frame.image_bgr.copy()
            )

            pose_landmarks_sets = (
                output.pose_result.pose_landmarks
            )

            hand_landmarks_sets = (
                output.hand_result.hand_landmarks
            )

            pose_count = len(
                pose_landmarks_sets
            )

            hand_count = len(
                hand_landmarks_sets
            )

            if measurement_started:
                frame_count += 1

                total_processing_s += (
                    output.processing_duration_s
                )

                if not validity.body_valid_for_control:
                    body_rejection_reasons.update(
                        validity.body_reasons
                    )

                if pose_count > 0:
                    pose_frame_count += 1

                if hand_count > 0:
                    hand_frame_count += 1

                total_hand_detections += (
                    hand_count
                )

                max_hands_seen = max(
                    max_hands_seen,
                    hand_count,
                )

                if hand_count >= 2:
                    two_hand_raw_frame_count += 1

                if observation.left_hands:
                    left_raw_detected_count += 1

                if observation.right_hands:
                    right_raw_detected_count += 1

                if not validity.left_hand_valid_for_control:
                    left_hand_rejection_reasons.update(
                        validity.left_hand_reasons
                    )

                if not validity.right_hand_valid_for_control:
                    right_hand_rejection_reasons.update(
                        validity.right_hand_reasons
                    )

                if (
                    validity
                    .available_for_visualization
                ):
                    visualization_frame_count += 1

                if (
                    validity
                    .body_valid_for_control
                ):
                    body_valid_count += 1

                if (
                    validity
                    .left_hand_valid_for_control
                ):
                    left_hand_valid_count += 1

                if (
                    validity
                    .right_hand_valid_for_control
                ):
                    right_hand_valid_count += 1

                if (
                    validity
                    .left_hand_valid_for_control
                    and validity
                    .right_hand_valid_for_control
                ):
                    both_hands_valid_count += 1

                body_state_counts[
                    tracking.body.state
                ] += 1

                left_state_counts[
                    tracking.left_hand.state
                ] += 1

                right_state_counts[
                    tracking.right_hand.state
                ] += 1

                if (
                    tracking
                    .body
                    .valid_for_control
                ):
                    body_temporal_valid_count += 1

                if (
                    tracking
                    .left_hand
                    .valid_for_control
                ):
                    left_temporal_valid_count += 1

                if (
                    tracking
                    .right_hand
                    .valid_for_control
                ):
                    right_temporal_valid_count += 1

                if (
                    tracking
                    .left_hand
                    .valid_for_control
                    and tracking
                    .right_hand
                    .valid_for_control
                ):
                    both_hands_temporal_valid_count += 1

                if smoothed.body.image_landmarks is not None:
                    smoothed_body_frame_count += 1

                    if observation.body_image_landmarks is not None:
                        separation_px = mean_xy_separation_px(
                            observation.body_image_landmarks.landmarks,
                            smoothed.body.image_landmarks.landmarks,
                            frame.image_width_px,
                            frame.image_height_px,
                        )

                        if separation_px is not None:
                            body_separation_sum_px += separation_px
                            body_separation_samples += 1

                if smoothed.left_hand.image_landmarks is not None:
                    smoothed_left_frame_count += 1

                    if len(observation.left_hands) == 1:
                        separation_px = mean_xy_separation_px(
                            observation.left_hands[0].image_landmarks.landmarks,
                            smoothed.left_hand.image_landmarks.landmarks,
                            frame.image_width_px,
                            frame.image_height_px,
                        )

                        if separation_px is not None:
                            left_separation_sum_px += separation_px
                            left_separation_samples += 1

                if smoothed.right_hand.image_landmarks is not None:
                    smoothed_right_frame_count += 1

                    if len(observation.right_hands) == 1:
                        separation_px = mean_xy_separation_px(
                            observation.right_hands[0].image_landmarks.landmarks,
                            smoothed.right_hand.image_landmarks.landmarks,
                            frame.image_width_px,
                            frame.image_height_px,
                        )

                        if separation_px is not None:
                            right_separation_sum_px += separation_px
                            right_separation_samples += 1


                if derived.body is not None:
                    derived_body_frame_count += 1

                    left_wrist = (
                        derived
                        .body
                        .left_wrist_displacement_normalized_body
                    )

                    right_wrist = (
                        derived
                        .body
                        .right_wrist_displacement_normalized_body
                    )

                    left_wrist_norm_sum += math.sqrt(
                        left_wrist.x * left_wrist.x
                        + left_wrist.y * left_wrist.y
                        + left_wrist.z * left_wrist.z
                    )

                    left_wrist_norm_samples += 1

                    right_wrist_norm_sum += math.sqrt(
                        right_wrist.x * right_wrist.x
                        + right_wrist.y * right_wrist.y
                        + right_wrist.z * right_wrist.z
                    )

                    right_wrist_norm_samples += 1

                else:
                    body_derived_reasons.update(
                        derived.body_reasons
                    )

                if derived.left_hand is not None:
                    derived_left_frame_count += 1

                    left_pinch = (
                        derived.left_hand.pinch_ratio
                    )

                    left_pinch_ratio_sum += left_pinch
                    left_pinch_ratio_samples += 1

                    left_pinch_ratio_min = (
                        left_pinch
                        if left_pinch_ratio_min is None
                        else min(
                            left_pinch_ratio_min,
                            left_pinch,
                        )
                    )

                    left_pinch_ratio_max = (
                        left_pinch
                        if left_pinch_ratio_max is None
                        else max(
                            left_pinch_ratio_max,
                            left_pinch,
                        )
                    )

                else:
                    left_derived_reasons.update(
                        derived.left_hand_reasons
                    )

                if derived.right_hand is not None:
                    derived_right_frame_count += 1

                    right_pinch = (
                        derived.right_hand.pinch_ratio
                    )

                    right_pinch_ratio_sum += right_pinch
                    right_pinch_ratio_samples += 1

                    right_pinch_ratio_min = (
                        right_pinch
                        if right_pinch_ratio_min is None
                        else min(
                            right_pinch_ratio_min,
                            right_pinch,
                        )
                    )

                    right_pinch_ratio_max = (
                        right_pinch
                        if right_pinch_ratio_max is None
                        else max(
                            right_pinch_ratio_max,
                            right_pinch,
                        )
                    )

                else:
                    right_derived_reasons.update(
                        derived.right_hand_reasons
                    )

                if retargeting_result is not None:
                    if retargeting_result.active:
                        retargeting_active_frame_count += 1

                        if retargeting_result.valid:
                            retargeting_valid_target_count += 1

                            if (
                                retargeting_result
                                .operator_delta_normalized_body
                                is not None
                            ):
                                delta = (
                                    retargeting_result
                                    .operator_delta_normalized_body
                                )

                                delta_norm = math.sqrt(
                                    delta.x * delta.x
                                    + delta.y * delta.y
                                    + delta.z * delta.z
                                )

                                max_operator_delta_norm = max(
                                    max_operator_delta_norm,
                                    delta_norm,
                                )

                            if (
                                retargeting_result
                                .target_pose
                                is not None
                            ):
                                target = (
                                    retargeting_result
                                    .target_pose
                                    .position
                                )

                                target_norm = math.sqrt(
                                    target.x * target.x
                                    + target.y * target.y
                                    + target.z * target.z
                                )

                                max_virtual_target_norm = max(
                                    max_virtual_target_norm,
                                    target_norm,
                                )

                            if (
                                retargeting_result
                                .workspace_limited
                            ):
                                retargeting_workspace_limited_count += 1

                            if (
                                retargeting_result
                                .speed_limited
                            ):
                                retargeting_speed_limited_count += 1

                        else:
                            retargeting_invalid_target_count += 1

                if logger is None:
                    raise RuntimeError(
                        "Measurement is active but H7/H8 logger "
                        "is not initialized."
                    )

                logger.write(
                    "frame",
                    {
                        "measurement_frame_index": (
                            frame_count - 1
                        ),
                        "frame_sequence_id": (
                            frame.sequence_id
                        ),
                        "measurement_timestamp_s": (
                            frame.measurement_timestamp_s
                        ),
                        "image_width_px": (
                            frame.image_width_px
                        ),
                        "image_height_px": (
                            frame.image_height_px
                        ),
                        "backend_processing_duration_s": (
                            output.processing_duration_s
                        ),
                        "raw_detection_counts": {
                            "pose": pose_count,
                            "hands": hand_count,
                        },
                        "observation": (
                            observation.to_dict()
                        ),
                        "per_frame_validity": {
                            "available_for_visualization": (
                                validity
                                .available_for_visualization
                            ),
                            "body_valid_for_control": (
                                validity
                                .body_valid_for_control
                            ),
                            "left_hand_valid_for_control": (
                                validity
                                .left_hand_valid_for_control
                            ),
                            "right_hand_valid_for_control": (
                                validity
                                .right_hand_valid_for_control
                            ),
                            "body_reasons": list(
                                validity.body_reasons
                            ),
                            "left_hand_reasons": list(
                                validity.left_hand_reasons
                            ),
                            "right_hand_reasons": list(
                                validity.right_hand_reasons
                            ),
                        },
                        "temporal_tracking": {
                            "body": (
                                tracking_channel_to_dict(
                                    tracking.body
                                )
                            ),
                            "left_hand": (
                                tracking_channel_to_dict(
                                    tracking.left_hand
                                )
                            ),
                            "right_hand": (
                                tracking_channel_to_dict(
                                    tracking.right_hand
                                )
                            ),
                        },
                        "smoothed_available": {
                            "body_image": (
                                smoothed
                                .body
                                .image_landmarks
                                is not None
                            ),
                            "body_world": (
                                smoothed
                                .body
                                .world_landmarks
                                is not None
                            ),
                            "left_hand_image": (
                                smoothed
                                .left_hand
                                .image_landmarks
                                is not None
                            ),
                            "left_hand_world": (
                                smoothed
                                .left_hand
                                .world_landmarks
                                is not None
                            ),
                            "right_hand_image": (
                                smoothed
                                .right_hand
                                .image_landmarks
                                is not None
                            ),
                            "right_hand_world": (
                                smoothed
                                .right_hand
                                .world_landmarks
                                is not None
                            ),
                        },
                        "smoothed": (
                            json_safe(
                                smoothed
                            )
                        ),
                        "derived": (
                            derived_to_dict(
                                derived
                            )
                        ),
                        "retargeting": (
                            retargeting_result_to_dict(
                                retargeting_result
                            )
                        ),
                    },
                )

            for pose_landmarks in (
                pose_landmarks_sets
            ):
                draw_landmark_set(
                    image=display,
                    landmarks=pose_landmarks,
                    connections=POSE_CONNECTIONS,
                    point_color=(0, 255, 0),
                    line_color=(255, 255, 255),
                )

            for (
                hand_index,
                hand_landmarks,
            ) in enumerate(
                hand_landmarks_sets
            ):
                hand_points = draw_landmark_set(
                    image=display,
                    landmarks=hand_landmarks,
                    connections=HAND_CONNECTIONS,
                    point_color=(0, 255, 255),
                    line_color=(255, 255, 255),
                )

                (
                    handedness_name,
                    handedness_score,
                ) = extract_handedness(
                    output.hand_result,
                    hand_index,
                )

                wrist_point = (
                    hand_points[0]
                    if hand_points
                    else None
                )

                if wrist_point is not None:
                    if handedness_score is None:
                        label = (
                            f"backend: "
                            f"{handedness_name}"
                        )

                    else:
                        label = (
                            f"backend: "
                            f"{handedness_name} "
                            f"{handedness_score:.2f}"
                        )

                    label_x = wrist_point[0]

                    label_y = max(
                        20,
                        wrist_point[1] - 15,
                    )

                    draw_text(
                        display,
                        label,
                        (label_x, label_y),
                        font_scale=0.50,
                        thickness=1,
                    )

            if smoothed.body.image_landmarks is not None:
                draw_landmark_set(
                    image=display,
                    landmarks=(
                        smoothed.body.image_landmarks.landmarks
                    ),
                    connections=POSE_CONNECTIONS,
                    point_color=SMOOTHED_COLOR,
                    line_color=SMOOTHED_COLOR,
                )

            if smoothed.left_hand.image_landmarks is not None:
                draw_landmark_set(
                    image=display,
                    landmarks=(
                        smoothed.left_hand.image_landmarks.landmarks
                    ),
                    connections=HAND_CONNECTIONS,
                    point_color=SMOOTHED_COLOR,
                    line_color=SMOOTHED_COLOR,
                )

            if smoothed.right_hand.image_landmarks is not None:
                draw_landmark_set(
                    image=display,
                    landmarks=(
                        smoothed.right_hand.image_landmarks.landmarks
                    ),
                    connections=HAND_CONNECTIONS,
                    point_color=SMOOTHED_COLOR,
                    line_color=SMOOTHED_COLOR,
                )

            processing_ms = (
                output.processing_duration_s
                * 1000.0
            )

            if (
                measurement_started
                and measurement_start_s is not None
            ):
                elapsed_s = (
                    time.perf_counter()
                    - measurement_start_s
                )
            else:
                elapsed_s = 0.0

            loop_fps = (
                frame_count / elapsed_s
                if elapsed_s > 0.0
                else 0.0
            )

            mode_text = (
                "MEASURING"
                if measurement_started
                else "PREVIEW - press S when ready"
            )

            draw_text(
                display,
                f"Mode: {mode_text}",
                (15, 25),
                font_scale=0.55,
                thickness=2,
            )

            draw_text(
                display,
                (
                    f"Frame {frame.sequence_id}  "
                    f"Pose: {pose_count}  "
                    f"Hands: {hand_count}"
                ),
                (15, 50),
                font_scale=0.55,
                thickness=2,
            )

            draw_text(
                display,
                (
                    f"Inference: "
                    f"{processing_ms:.1f} ms  "
                    f"Loop: {loop_fps:.1f} FPS"
                ),
                (15, 75),
                font_scale=0.50,
                thickness=2,
            )

            body_quality = (
                "OK"
                if validity.body_valid_for_control
                else "BLOCK"
            )

            left_quality = (
                "OK"
                if validity.left_hand_valid_for_control
                else "BLOCK"
            )

            right_quality = (
                "OK"
                if validity.right_hand_valid_for_control
                else "BLOCK"
            )

            draw_text(
                display,
                (
                    "Frame quality - "
                    f"B:{body_quality} "
                    f"L:{left_quality} "
                    f"R:{right_quality}"
                ),
                (15, 100),
                font_scale=0.50,
                thickness=2,
            )

            draw_text(
                display,
                (
                    "Temporal - "
                    f"B:{tracking.body.state.value.upper()} "
                    f"L:{tracking.left_hand.state.value.upper()} "
                    f"R:{tracking.right_hand.state.value.upper()}"
                ),
                (15, 125),
                font_scale=0.47,
                thickness=2,
            )

            body_control = (
                "GO"
                if tracking.body.valid_for_control
                else "HOLD"
            )

            left_control = (
                "GO"
                if tracking.left_hand.valid_for_control
                else "HOLD"
            )

            right_control = (
                "GO"
                if tracking.right_hand.valid_for_control
                else "HOLD"
            )

            draw_text(
                display,
                (
                    "Temporal control - "
                    f"B:{body_control} "
                    f"L:{left_control} "
                    f"R:{right_control}"
                ),
                (15, 150),
                font_scale=0.47,
                thickness=2,
            )

            smoothing_status = (
                "Smoothing - "
                f"B:{'ON' if smoothed.body.image_landmarks is not None else 'OFF'} "
                f"L:{'ON' if smoothed.left_hand.image_landmarks is not None else 'OFF'} "
                f"R:{'ON' if smoothed.right_hand.image_landmarks is not None else 'OFF'}"
            )

            draw_text(
                display,
                smoothing_status,
                (15, 175),
                font_scale=0.47,
                thickness=2,
            )

            body_derived_status = (
                "ON"
                if derived.body is not None
                else "OFF"
            )

            if derived.left_hand is not None:
                left_pinch_text = (
                    f"{derived.left_hand.pinch_ratio:.2f}"
                )
            else:
                left_pinch_text = "--"

            if derived.right_hand is not None:
                right_pinch_text = (
                    f"{derived.right_hand.pinch_ratio:.2f}"
                )
            else:
                right_pinch_text = "--"

            draw_text(
                display,
                (
                    "H6 - "
                    f"Body:{body_derived_status} "
                    f"L pinch:{left_pinch_text} "
                    f"R pinch:{right_pinch_text}"
                ),
                (15, 200),
                font_scale=0.45,
                thickness=2,
            )

            if derived.body is not None:
                left_wrist = (
                    derived
                    .body
                    .left_wrist_displacement_normalized_body
                )

                right_wrist = (
                    derived
                    .body
                    .right_wrist_displacement_normalized_body
                )

                draw_text(
                    display,
                    (
                        "H6 wrist/body - "
                        f"L({left_wrist.x:+.2f},"
                        f"{left_wrist.y:+.2f},"
                        f"{left_wrist.z:+.2f}) "
                        f"R({right_wrist.x:+.2f},"
                        f"{right_wrist.y:+.2f},"
                        f"{right_wrist.z:+.2f})"
                    ),
                    (15, 220),
                    font_scale=0.40,
                    thickness=1,
                )

            if not measurement_started:
                h8_status = (
                    "H8 - wait for S"
                )
            elif not retargeter.active:
                h8_status = (
                    f"H8 {control_arm.value.upper()} - "
                    "INACTIVE, press R"
                )
            elif (
                retargeting_result is not None
                and retargeting_result.valid
                and retargeting_result.target_pose is not None
            ):
                target = (
                    retargeting_result
                    .target_pose
                    .position
                )

                h8_status = (
                    f"H8 {control_arm.value.upper()} target "
                    f"({target.x:+.2f},"
                    f"{target.y:+.2f},"
                    f"{target.z:+.2f})"
                )
            else:
                h8_status = (
                    f"H8 {control_arm.value.upper()} - "
                    "NO TARGET"
                )

            draw_text(
                display,
                h8_status,
                (15, 315),
                font_scale=0.45,
                thickness=2,
            )

            if (
                retargeting_result is not None
                and retargeting_result.valid
                and retargeting_result
                .operator_delta_normalized_body
                is not None
            ):
                delta = (
                    retargeting_result
                    .operator_delta_normalized_body
                )

                draw_text(
                    display,
                    (
                        "H8 operator delta/body - "
                        f"({delta.x:+.2f},"
                        f"{delta.y:+.2f},"
                        f"{delta.z:+.2f})"
                    ),
                    (15, 335),
                    font_scale=0.40,
                    thickness=1,
                )

            if (
                not tracking
                .body
                .valid_for_control
            ):
                draw_text(
                    display,
                    (
                        "B reason: "
                        f"{first_reason(tracking.body.reasons)}"
                    ),
                    (15, 245),
                    font_scale=0.42,
                    thickness=1,
                )

            if (
                not tracking
                .left_hand
                .valid_for_control
            ):
                draw_text(
                    display,
                    (
                        "L reason: "
                        f"{first_reason(tracking.left_hand.reasons)}"
                    ),
                    (15, 265),
                    font_scale=0.42,
                    thickness=1,
                )

            if (
                not tracking
                .right_hand
                .valid_for_control
            ):
                draw_text(
                    display,
                    (
                        "R reason: "
                        f"{first_reason(tracking.right_hand.reasons)}"
                    ),
                    (15, 285),
                    font_scale=0.42,
                    thickness=1,
                )

            cv2.imshow(
                WINDOW_NAME,
                display,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if (
                not measurement_started
                and key in (
                    ord("s"),
                    ord("S"),
                )
            ):
                tracker = HumanTemporalTracker(
                    temporal_config
                )

                smoother = HumanLandmarkSmoother(
                    smoothing_config
                )

                retargeter = RelativeRetargeter(
                    arm_side=control_arm,
                    config=h8_config,
                )

                retargeting_result = None

                logger = JsonlDiagnosticLogger(
                    args.log_dir
                )

                log_path = logger.path

                measurement_started = True
                measurement_start_s = time.perf_counter()

                logger.write(
                    "run_start",
                    {
                        "created_utc": (
                            datetime.now(timezone.utc)
                            .isoformat()
                        ),
                        "camera_index": (
                            args.camera
                        ),
                        "camera_reported_properties": (
                            acquisition
                            .get_reported_properties()
                        ),
                        "runtime": {
                            "python_version": (
                                sys.version.split()[0]
                            ),
                            "platform": (
                                platform.platform()
                            ),
                            "mediapipe_version": (
                                mp.__version__
                            ),
                            "opencv_version": (
                                cv2.__version__
                            ),
                        },
                        "pose_model": (
                            file_identity(
                                args.pose_model
                            )
                        ),
                        "hand_model": (
                            file_identity(
                                args.hand_model
                            )
                        ),
                        "validation_config": {
                            "min_body_visibility": (
                                validation_config
                                .min_body_visibility
                            ),
                            "min_body_presence": (
                                validation_config
                                .min_body_presence
                            ),
                            "min_handedness_score": (
                                validation_config
                                .min_handedness_score
                            ),
                        },
                        "temporal_config": {
                            "consecutive_valid_required": (
                                temporal_config
                                .consecutive_valid_required
                            ),
                            "dropout_timeout_s": (
                                temporal_config
                                .dropout_timeout_s
                            ),
                        },
                        "smoothing_config": {
                            "time_constant_s": (
                                smoothing_config
                                .time_constant_s
                            ),
                        },
                        "derived_geometry_config": {
                            "min_length_model_world": (
                                derived_config
                                .min_length_model_world
                            ),
                        },
                        "retargeting_config": {
                            "diagnostic_only": True,
                            "control_arm": (
                                control_arm.value
                            ),
                            "axis_mapping": (
                                h8_config
                                .axis_mapping
                                .rows
                            ),
                            "scale_robot_per_normalized_body": (
                                h8_config
                                .scale_robot_per_normalized_body
                            ),
                            "deadband_normalized_body": (
                                h8_config
                                .deadband_normalized_body
                            ),
                            "workspace_bounds": (
                                h8_config
                                .workspace_bounds
                            ),
                            "max_cartesian_speed_robot_per_s": (
                                h8_config
                                .max_cartesian_speed_robot_per_s
                            ),
                            "virtual_reference_pose": (
                                h8_virtual_reference_pose
                            ),
                            "units_note": (
                                "H8 live validation uses virtual unitless "
                                "target coordinates. These are not physical "
                                "robot workspace units."
                            ),
                        },
                        "coordinate_note": (
                            "MEDIAPIPE_WORLD values are model-world "
                            "geometry, not the calibrated Part 1 workspace."
                        ),
                        "logging_note": (
                            "Each measured frame stores raw observation, "
                            "validity, temporal state, full smoothed "
                            "landmarks, derived geometry, H8 retargeting "
                            "state, and timing."
                        ),
                        "display_mirrored": False,
                        "handedness_note": (
                            "Raw MediaPipe Left/Right was physically "
                            "verified to match anatomical left/right "
                            "with the current unmirrored camera pipeline."
                        ),
                    },
                )

                print(
                    "Measurement started. "
                    "Only frames from this point onward "
                    "will be included in the statistics."
                )

                print(
                    "H7 log: "
                    f"{log_path}"
                )

                continue

            if (
                measurement_started
                and key in (
                    ord("r"),
                    ord("R"),
                )
            ):
                if derived.body is None:
                    print(
                        "H8 reference NOT captured: "
                        "derived body geometry is unavailable."
                    )
                else:
                    reference = retargeter.activate(
                        derived,
                        h8_virtual_reference_pose,
                    )

                    retargeting_activation_count += 1

                    if logger is not None:
                        logger.write(
                            "retargeting_reference",
                            {
                                "activation_index": (
                                    retargeting_activation_count
                                ),
                                "arm_side": (
                                    reference.arm_side.value
                                ),
                                "activation_frame_sequence_id": (
                                    reference
                                    .activation_frame_sequence_id
                                ),
                                "activation_timestamp_s": (
                                    reference
                                    .activation_timestamp_s
                                ),
                                "human_wrist_reference_normalized_body": (
                                    reference
                                    .human_wrist_reference_normalized_body
                                ),
                                "robot_pose_reference": (
                                    reference.robot_pose_reference
                                ),
                            },
                        )

                    print(
                        "H8 reference captured for "
                        f"{control_arm.value} arm at "
                        f"frame {reference.activation_frame_sequence_id}. "
                        "Virtual target is now relative to this pose."
                    )

                continue

            if key in (
                ord("q"),
                ord("Q"),
            ):
                break

    except Exception as exc:
        print()
        print("RESULT: FAIL")

        print(
            f"{type(exc).__name__}: {exc}"
        )

        if logger is not None:
            try:
                logger.write(
                    "run_error",
                    {
                        "error_type": (
                            type(exc).__name__
                        ),
                        "error_message": str(exc),
                    },
                )
            except Exception:
                pass

            logger.close()

        raise SystemExit(1)

    finally:
        backend.close()
        acquisition.close()
        cv2.destroyAllWindows()

    if (
        measurement_started
        and measurement_start_s is not None
    ):
        duration_s = (
            time.perf_counter()
            - measurement_start_s
        )
    else:
        duration_s = 0.0

    average_processing_ms = (
        1000.0
        * total_processing_s
        / frame_count
        if frame_count > 0
        else 0.0
    )

    average_loop_fps = (
        frame_count / duration_s
        if duration_s > 0.0
        else 0.0
    )

    print()
    print("-" * 60)

    if not measurement_started or frame_count == 0:
        print("No measurement was recorded.")
        print(
            "Run the test again and press S after "
            "you are in the intended test position."
        )
        raise SystemExit(0)

    print(f"Frames processed: {frame_count}")

    print(
        "Frames with raw pose: "
        f"{pose_frame_count} "
        f"({percentage(pose_frame_count, frame_count):.1f}%)"
    )

    print(
        "Frames with >=1 raw hand: "
        f"{hand_frame_count} "
        f"({percentage(hand_frame_count, frame_count):.1f}%)"
    )

    print(
        "Total raw hand detections: "
        f"{total_hand_detections}"
    )

    print(
        "Maximum simultaneous raw hands: "
        f"{max_hands_seen}"
    )

    print(
        "Frames with 2 raw hands: "
        f"{two_hand_raw_frame_count} "
        f"({percentage(two_hand_raw_frame_count, frame_count):.1f}%)"
    )

    print(
        "Frames with raw Left hand: "
        f"{left_raw_detected_count} "
        f"({percentage(left_raw_detected_count, frame_count):.1f}%)"
    )

    print(
        "Frames with raw Right hand: "
        f"{right_raw_detected_count} "
        f"({percentage(right_raw_detected_count, frame_count):.1f}%)"
    )

    print()

    print(
        "Available for visualization: "
        f"{visualization_frame_count} "
        f"({percentage(visualization_frame_count, frame_count):.1f}%)"
    )

    print(
        "Body per-frame control-valid: "
        f"{body_valid_count} "
        f"({percentage(body_valid_count, frame_count):.1f}%)"
    )

    print(
        "Left-hand per-frame control-valid: "
        f"{left_hand_valid_count} "
        f"({percentage(left_hand_valid_count, frame_count):.1f}%)"
    )

    print(
        "Right-hand per-frame control-valid: "
        f"{right_hand_valid_count} "
        f"({percentage(right_hand_valid_count, frame_count):.1f}%)"
    )

    print(
        "Both hands per-frame control-valid: "
        f"{both_hands_valid_count} "
        f"({percentage(both_hands_valid_count, frame_count):.1f}%)"
    )

    print()
    print("After temporal gating:")

    print(
        "Body temporal control-valid: "
        f"{body_temporal_valid_count} "
        f"({percentage(body_temporal_valid_count, frame_count):.1f}%)"
    )

    print(
        "Left temporal control-valid: "
        f"{left_temporal_valid_count} "
        f"({percentage(left_temporal_valid_count, frame_count):.1f}%)"
    )

    print(
        "Right temporal control-valid: "
        f"{right_temporal_valid_count} "
        f"({percentage(right_temporal_valid_count, frame_count):.1f}%)"
    )

    print(
        "Both hands temporal control-valid: "
        f"{both_hands_temporal_valid_count} "
        f"({percentage(both_hands_temporal_valid_count, frame_count):.1f}%)"
    )

    print()

    print(
        "Body temporal states: "
        f"{format_state_counts(body_state_counts)}"
    )

    print(
        "Left temporal states: "
        f"{format_state_counts(left_state_counts)}"
    )

    print(
        "Right temporal states: "
        f"{format_state_counts(right_state_counts)}"
    )

    print()
    print("Body rejection reasons:")

    if body_rejection_reasons:
        for reason, count in (
            body_rejection_reasons.most_common()
        ):
            print(
                f"  {reason}: "
                f"{count} "
                f"({percentage(count, frame_count):.1f}% of frames)"
            )
    else:
        print("  none")

    print()
    print("Left-hand rejection reasons:")

    if left_hand_rejection_reasons:
        for reason, count in (
            left_hand_rejection_reasons.most_common()
        ):
            print(
                f"  {reason}: "
                f"{count} "
                f"({percentage(count, frame_count):.1f}% of frames)"
            )
    else:
        print("  none")

    print()
    print("Right-hand rejection reasons:")

    if right_hand_rejection_reasons:
        for reason, count in (
            right_hand_rejection_reasons.most_common()
        ):
            print(
                f"  {reason}: "
                f"{count} "
                f"({percentage(count, frame_count):.1f}% of frames)"
            )
    else:
        print("  none")

    print()
    print("H5 smoothing summary:")

    print(
        "Smoothed body frames: "
        f"{smoothed_body_frame_count} "
        f"({percentage(smoothed_body_frame_count, frame_count):.1f}%)"
    )

    print(
        "Smoothed Left-hand frames: "
        f"{smoothed_left_frame_count} "
        f"({percentage(smoothed_left_frame_count, frame_count):.1f}%)"
    )

    print(
        "Smoothed Right-hand frames: "
        f"{smoothed_right_frame_count} "
        f"({percentage(smoothed_right_frame_count, frame_count):.1f}%)"
    )

    mean_body_separation_px = (
        body_separation_sum_px / body_separation_samples
        if body_separation_samples > 0
        else 0.0
    )

    mean_left_separation_px = (
        left_separation_sum_px / left_separation_samples
        if left_separation_samples > 0
        else 0.0
    )

    mean_right_separation_px = (
        right_separation_sum_px / right_separation_samples
        if right_separation_samples > 0
        else 0.0
    )

    print(
        "Mean raw-to-smoothed body separation: "
        f"{mean_body_separation_px:.2f} px"
    )

    print(
        "Mean raw-to-smoothed Left-hand separation: "
        f"{mean_left_separation_px:.2f} px"
    )

    print(
        "Mean raw-to-smoothed Right-hand separation: "
        f"{mean_right_separation_px:.2f} px"
    )

    print(
        "The separation values describe smoothing deviation; "
        "they are not direct latency measurements."
    )

    print()
    print("H6 derived-geometry summary:")

    print(
        "Derived body frames: "
        f"{derived_body_frame_count} "
        f"({percentage(derived_body_frame_count, frame_count):.1f}%)"
    )

    print(
        "Derived Left-hand frames: "
        f"{derived_left_frame_count} "
        f"({percentage(derived_left_frame_count, frame_count):.1f}%)"
    )

    print(
        "Derived Right-hand frames: "
        f"{derived_right_frame_count} "
        f"({percentage(derived_right_frame_count, frame_count):.1f}%)"
    )

    if left_wrist_norm_samples > 0:
        print(
            "Mean normalized Left wrist displacement magnitude: "
            f"{left_wrist_norm_sum / left_wrist_norm_samples:.3f}"
        )

    if right_wrist_norm_samples > 0:
        print(
            "Mean normalized Right wrist displacement magnitude: "
            f"{right_wrist_norm_sum / right_wrist_norm_samples:.3f}"
        )

    if left_pinch_ratio_samples > 0:
        print(
            "Left pinch ratio mean/min/max: "
            f"{left_pinch_ratio_sum / left_pinch_ratio_samples:.3f} / "
            f"{left_pinch_ratio_min:.3f} / "
            f"{left_pinch_ratio_max:.3f}"
        )

    if right_pinch_ratio_samples > 0:
        print(
            "Right pinch ratio mean/min/max: "
            f"{right_pinch_ratio_sum / right_pinch_ratio_samples:.3f} / "
            f"{right_pinch_ratio_min:.3f} / "
            f"{right_pinch_ratio_max:.3f}"
        )

    print()
    print("H6 body derived-geometry rejection reasons:")

    if body_derived_reasons:
        for reason, count in (
            body_derived_reasons.most_common()
        ):
            print(
                f"  {reason}: "
                f"{count} "
                f"({percentage(count, frame_count):.1f}% of frames)"
            )
    else:
        print("  none")

    print()
    print("H6 Left-hand derived-geometry rejection reasons:")

    if left_derived_reasons:
        for reason, count in (
            left_derived_reasons.most_common()
        ):
            print(
                f"  {reason}: "
                f"{count} "
                f"({percentage(count, frame_count):.1f}% of frames)"
            )
    else:
        print("  none")

    print()
    print("H6 Right-hand derived-geometry rejection reasons:")

    if right_derived_reasons:
        for reason, count in (
            right_derived_reasons.most_common()
        ):
            print(
                f"  {reason}: "
                f"{count} "
                f"({percentage(count, frame_count):.1f}% of frames)"
            )
    else:
        print("  none")

    print()

    print("H8 virtual-retargeting summary:")

    print(
        "Reference captures: "
        f"{retargeting_activation_count}"
    )

    print(
        "Active retargeting frames: "
        f"{retargeting_active_frame_count} "
        f"({percentage(retargeting_active_frame_count, frame_count):.1f}%)"
    )

    print(
        "Valid virtual target frames: "
        f"{retargeting_valid_target_count} "
        f"({percentage(retargeting_valid_target_count, frame_count):.1f}%)"
    )

    print(
        "Invalid target frames while active: "
        f"{retargeting_invalid_target_count}"
    )

    print(
        "Workspace-limited frames: "
        f"{retargeting_workspace_limited_count}"
    )

    print(
        "Speed-limited frames: "
        f"{retargeting_speed_limited_count}"
    )

    print(
        "Maximum normalized operator delta magnitude: "
        f"{max_operator_delta_norm:.3f}"
    )

    print(
        "Maximum virtual target displacement magnitude: "
        f"{max_virtual_target_norm:.3f}"
    )

    print(
        "H8 live coordinates are diagnostic virtual units, "
        "not physical robot coordinates."
    )

    print()

    print(
        "Average backend processing: "
        f"{average_processing_ms:.2f} ms"
    )

    print(
        "Average complete loop rate: "
        f"{average_loop_fps:.2f} FPS"
    )

    print("-" * 60)

    raw_pipeline_ok = (
        frame_count > 0
        and pose_frame_count > 0
        and max_hands_seen >= 2
    )

    temporal_tracking_seen = (
        body_state_counts[
            TrackingState.TRACKING
        ] > 0
        and left_state_counts[
            TrackingState.TRACKING
        ] > 0
        and right_state_counts[
            TrackingState.TRACKING
        ] > 0
    )

    hand_dropout_seen = (
        left_state_counts[
            TrackingState.DROPOUT
        ] > 0
        or right_state_counts[
            TrackingState.DROPOUT
        ] > 0
    )

    hand_lost_seen = (
        left_state_counts[
            TrackingState.LOST
        ] > 0
        or right_state_counts[
            TrackingState.LOST
        ] > 0
    )

    print(
        "Observed hand DROPOUT state: "
        f"{'YES' if hand_dropout_seen else 'NO'}"
    )

    print(
        "Observed hand LOST state: "
        f"{'YES' if hand_lost_seen else 'NO'}"
    )

    smoothing_seen = (
        smoothed_body_frame_count > 0
        and smoothed_left_frame_count > 0
        and smoothed_right_frame_count > 0
    )

    derived_seen = (
        derived_body_frame_count > 0
        and derived_left_frame_count > 0
        and derived_right_frame_count > 0
    )

    if logger is not None:
        logger.write(
            "run_summary",
            {
                "duration_s": duration_s,
                "frames_processed": frame_count,
                "raw_pose_frames": pose_frame_count,
                "raw_hand_frames": hand_frame_count,
                "two_raw_hand_frames": (
                    two_hand_raw_frame_count
                ),
                "body_per_frame_valid": (
                    body_valid_count
                ),
                "left_hand_per_frame_valid": (
                    left_hand_valid_count
                ),
                "right_hand_per_frame_valid": (
                    right_hand_valid_count
                ),
                "body_temporal_valid": (
                    body_temporal_valid_count
                ),
                "left_hand_temporal_valid": (
                    left_temporal_valid_count
                ),
                "right_hand_temporal_valid": (
                    right_temporal_valid_count
                ),
                "smoothed_body_frames": (
                    smoothed_body_frame_count
                ),
                "smoothed_left_hand_frames": (
                    smoothed_left_frame_count
                ),
                "smoothed_right_hand_frames": (
                    smoothed_right_frame_count
                ),
                "derived_body_frames": (
                    derived_body_frame_count
                ),
                "derived_left_hand_frames": (
                    derived_left_frame_count
                ),
                "derived_right_hand_frames": (
                    derived_right_frame_count
                ),
                "retargeting_activation_count": (
                    retargeting_activation_count
                ),
                "retargeting_active_frames": (
                    retargeting_active_frame_count
                ),
                "retargeting_valid_target_frames": (
                    retargeting_valid_target_count
                ),
                "retargeting_invalid_target_frames": (
                    retargeting_invalid_target_count
                ),
                "retargeting_workspace_limited_frames": (
                    retargeting_workspace_limited_count
                ),
                "retargeting_speed_limited_frames": (
                    retargeting_speed_limited_count
                ),
                "retargeting_max_operator_delta_norm": (
                    max_operator_delta_norm
                ),
                "retargeting_max_virtual_target_norm": (
                    max_virtual_target_norm
                ),
                "average_backend_processing_ms": (
                    average_processing_ms
                ),
                "average_complete_loop_fps": (
                    average_loop_fps
                ),
                "body_rejection_reasons": (
                    dict(body_rejection_reasons)
                ),
                "left_hand_rejection_reasons": (
                    dict(left_hand_rejection_reasons)
                ),
                "right_hand_rejection_reasons": (
                    dict(right_hand_rejection_reasons)
                ),
                "body_derived_reasons": (
                    dict(body_derived_reasons)
                ),
                "left_derived_reasons": (
                    dict(left_derived_reasons)
                ),
                "right_derived_reasons": (
                    dict(right_derived_reasons)
                ),
            },
        )

        logger.close()

    if log_path is not None:
        print(
            "H7 JSONL log saved to: "
            f"{log_path}"
        )

    h8_seen = (
        retargeting_activation_count > 0
        and retargeting_valid_target_count > 0
    )

    if (
        raw_pipeline_ok
        and temporal_tracking_seen
        and smoothing_seen
        and derived_seen
        and h8_seen
        and log_path is not None
    ):
        print(
            "LIVE H8 RESULT: PASS"
        )

        print(
            "Relative virtual retargeting produced valid "
            "robot-agnostic targets from the live H6 stream."
        )

    else:
        print(
            "LIVE H8 RESULT: INCOMPLETE"
        )

        print(
            "The pipeline ran, but H8 was not activated or "
            "did not produce a valid virtual target."
        )


if __name__ == "__main__":
    main()
