from __future__ import annotations
import argparse
from dataclasses import dataclass, field
from collections import Counter
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
import platform
import statistics
from pathlib import Path
import sys
import time
import cv2
import mediapipe as mp
from human_tracking.acquisition import CameraAcquisition
from human_tracking.mediapipe_adapter import MediaPipeObservationAdapter
from human_tracking.mediapipe_backend import MediaPipeTrackingBackend
from human_tracking.observation import (
    ControlArm,
    HandAssociationSource,
    ObservationValidationConfig,
    associate_hands_to_pose,
    evaluate_observation
)
from human_tracking.tracking import (
    DerivedGeometryConfig,
    HumanLandmarkSmoother,
    HumanTemporalTracker,
    LandmarkSmoothingConfig,
    TemporalTrackingConfig,
    TrackingState,
    Vector3,
    derive_human_kinematics
)
from human_tracking.retargeting import (
    ArmSide,
    AxisMapping,
    CartesianBounds,
    Quaternion,
    RelativeRetargeter,
    RetargetingConfig,
    RobotAgnosticPose
)
from human_tracking.gripper_intent import (
    GripperApertureConfig,
    GripperApertureResult,
    GripperApertureTracker,
    HandSide
)
from human_tracking.supervisor import (
    CommandSupervisor,
    CommandSupervisorConfig,
    CommandSupervisorResult,
    GripperLossPolicy,
    SupervisorCycleInput,
    SupervisorState
)

WINDOW_NAME = "H2/H4/H5/H6/H8/P3 - Human Tracking and Pinch Calibration"

# OpenCV uses BGR color ordering.
# Equivalent RGB color: #39FF14.
TEXT_COLOR = (20, 255, 57)
TEXT_OUTLINE_COLOR = (0, 0, 0)
SMOOTHED_COLOR = (255, 0, 255)

POSE_CONNECTIONS = ((0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
    (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32),
    (27, 31), (28, 32)
)

HAND_CONNECTIONS = ((0, 1), (1, 5), (5, 9), (9, 13), (13, 17), (17, 0), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20)
)

def parse_args():
    parser = argparse.ArgumentParser(
        description = ("Live MediaPipe Pose + Hand tracking with H4 per-frame and temporal validity gates.")
    )

    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index. Default: 0")
    parser.add_argument("--pose-model", type=Path, default = Path("models/pose_landmarker_full.task"))
    parser.add_argument("--hand-model", type=Path, default = Path("models/hand_landmarker.task"))
    parser.add_argument("--hand-min-detection-confidence", type=float, default=0.50,
        help = ("MediaPipe minimum palm-detection confidence. Diagnostic tuning parameter in [0, 1]. Default: 0.50")
    )
    parser.add_argument("--hand-min-presence-confidence", type=float, default=0.50,
        help = ("MediaPipe minimum hand-presence confidence. Diagnostic tuning parameter in [0, 1]. Default: 0.50")
    )
    parser.add_argument("--hand-min-tracking-confidence", type=float, default=0.50,
        help = ("MediaPipe minimum hand-tracking confidence/IoU threshold. Diagnostic tuning parameter in [0, 1]. Default: 0.50")
    )
    parser.add_argument("--hand-min-palm-width-to-palm-length-ratio", type=float, default=None,
        help = ("Optional H6 hand-geometry plausibility gate. The value is the minimum MediaPipe model-world palm width (index MCP to pinky MCP)"
            " divided by palm length (wrist to middle MCP). No default is applied; 0.27 is the current data-derived HT3 test value.")
    )
    parser.add_argument("--smoothing-time-constant", type=float, default=0.10,
        help = ("Low-pass smoothing time constant in seconds. Default: 0.10")
    )
    parser.add_argument("--log-dir", type=Path, default=Path("logs"),
        help = ("Directory for H7/H8 JSONL diagnostic logs. Default: logs")
    )
    parser.add_argument("--control-arm", choices=("left", "right"), default="right",
        help = ("Arm used by the H8 virtual retargeting check. Default: right")
    )
    parser.add_argument("--gesture-hand", choices=("left", "right"), default=None,
        help = ("Hand used by the P3 pinch-calibration check and, when configured, by the P4E continuous-aperture channel. Defaults to --control-arm.")
    )
    parser.add_argument("--hand-association", choices=("backend", "pose_wrist"), default="pose_wrist",
        help = ("Hand-to-anatomical-side association mode. 'pose_wrist' uses current-frame pose wrists with a shoulder-width-normalized gate; "
            "'backend' preserves raw MediaPipe handedness buckets. Default: pose_wrist"
        )
    )
    parser.add_argument("--hand-association-max-distance-shoulder-widths", type=float, default=0.50,
        help = ("Maximum image-space hand-wrist to pose-wrist distance, normalized by pose shoulder width, for pose-guided association."
            " The 0.50 default is provisional and data-derived; it is not a physical robot/workspace threshold."
        )
    )
    parser.add_argument("--closed-reference-ratio", type=float, default=None,
        help = ("Explicit H6 palm-length-normalized thumb-index pinch ratio representing the comfortably closed endpoint for P4E."
            " No default is supplied."
        )
    )
    parser.add_argument("--open-reference-ratio", type=float, default=None,
        help = ("Explicit H6 palm-length-normalized thumb-index pinch ratio representing the comfortably open endpoint for P4E."
            " No default is supplied."
        )
    )
    parser.add_argument("--gripper-loss-policy", choices = (GripperLossPolicy.HOLD_TELEOP.value, GripperLossPolicy.ALLOW_ARM_ONLY.value), default = None,
        help = ("Explicit P4E supervisor policy for gripper-hand loss. Required when aperture references are supplied.")
    )
    parser.add_argument("--p4e-auto-sequence", action="store_true",
        help = ("Run the P4E interaction hands-free after launch. Measurement starts after --p4e-auto-start-delay-s, H8 reference capture "
            "waits for current valid body+aperture data, teleoperation is requested automatically, the initial READY state is explicitly "
            "enabled by the scripted diagnostic, and a post-ACTIVE diagnostic window runs for --p4e-auto-active-duration-s before disabling and "
            "quitting. HOLD remains fail-closed but no longer ends the window; the script never automatically re-enables after HOLD. Press Q in "
            "the camera window to stop early."
        )
    )
    parser.add_argument("--p4e-auto-start-delay-s", type=float, default=10.0,
        help = ("Operator-positioning countdown before automatic measurement start."
            " Diagnostic UI timing only; not a robot/control limit. Default: 10.0 s"
        )
    )
    parser.add_argument("--p4e-auto-active-duration-s", type=float, default=90.0,
        help = ("Length of the hands-free diagnostic window after the supervisor first reaches ACTIVE."
            " HOLD remains fail-closed and does not stop the timer or trigger automatic re-enable. Press Q in the camera "
            "window to stop early. Diagnostic UI timing only; not a robot/control limit. Default: 90.0 s"
        )
    )

    return parser.parse_args()

def landmark_to_pixel(landmark, width: int, height: int):
    x = float(landmark.x)
    y = float(landmark.y)

    if not (math.isfinite(x) and math.isfinite(y)):
        return None

    if x < 0.0 or x > 1.0:
        return None

    if y < 0.0 or y > 1.0:
        return None

    px = int(round(x * (width - 1)))
    py = int(round(y * (height - 1)))

    return px, py

def draw_landmark_set(image, landmarks, connections, point_color, line_color):
    height, width = image.shape[:2]

    points = [
        landmark_to_pixel(landmark, width, height)
        for landmark in landmarks
    ]

    for start, end in connections:
        if start >= len(points):
            continue

        if end >= len(points):
            continue

        start_point = points[start]
        end_point = points[end]

        if start_point is None or end_point is None:
            continue

        cv2.line(image, start_point, end_point, line_color, 2, cv2.LINE_AA)

    for point in points:
        if point is None:
            continue

        cv2.circle(image, point, 3, point_color, -1, cv2.LINE_AA)

    return points

def draw_text(
    image,
    text: str,
    position: tuple[int, int],
    font_scale: float = 0.50,
    thickness: int = 1
):

    cv2.putText(
        image, text, position,
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, TEXT_OUTLINE_COLOR, thickness + 3,
        cv2.LINE_AA
    )

    cv2.putText(
        image, text, position,
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, TEXT_COLOR, thickness,
        cv2.LINE_AA
    )

def mean_xy_separation_px(raw_landmarks, smoothed_landmarks, width: int, height: int) -> float | None:
    if len(raw_landmarks) != len(smoothed_landmarks):
        return None

    distances = []

    for raw, smoothed in zip(raw_landmarks, smoothed_landmarks):
        raw_x = float(raw.x)
        raw_y = float(raw.y)
        smooth_x = float(smoothed.x)
        smooth_y = float(smoothed.y)

        if not all(
            math.isfinite(value)
            for value in (raw_x, raw_y, smooth_x, smooth_y)
        ):
            continue

        dx_px = (raw_x - smooth_x) * width
        dy_px = (raw_y - smooth_y) * height

        distances.append(math.hypot(dx_px, dy_px))

    if not distances:
        return None

    return sum(distances) / len(distances)

def extract_handedness(hand_result, hand_index: int):
    if hand_index >= len(hand_result.handedness):
        return "Unknown", None

    categories = hand_result.handedness[hand_index]

    if not categories:
        return "Unknown", None

    category = categories[0]
    name = getattr(category, "category_name", None)
    score = getattr(category,"score",None)

    if not name:
        name = "Unknown"

    if score is not None:
        score = float(score)

    return str(name), score

def first_reason(reasons: tuple[str, ...]) -> str:
    if not reasons:
        return "none"

    return reasons[0]

def percentage(count: int, total: int) -> float:
    if total <= 0:
        return 0.0

    return 100.0 * count / total

def format_state_counts(counts: dict[TrackingState, int]) -> str:
    return ", ".join(
        f"{state.value}={counts[state]}"
        for state in TrackingState
    )

LOG_SCHEMA_NAME = "human_tracking_diagnostic"
LOG_SCHEMA_VERSION = 9

def json_safe(value):
    if value is None:
        return None

    if isinstance(value, ( str, int, float, bool)):
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
        "current_frame_valid": channel.current_frame_valid,
        "valid_for_control": channel.valid_for_control,
        "consecutive_valid_frames": channel.consecutive_valid_frames,
        "age_since_last_valid_s": channel.age_since_last_valid_s,
        "reasons": list(channel.reasons)
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
        "frame_sequence_id": derived.frame_sequence_id,
        "measurement_timestamp_s": derived.measurement_timestamp_s,
        "body": None,
        "left_hand": None,
        "right_hand": None,
        "body_reasons": list(derived.body_reasons),
        "left_hand_reasons": list(derived.left_hand_reasons),
        "right_hand_reasons": list(derived.right_hand_reasons)
    }

    if derived.body is not None:
        body = derived.body

        result["body"] = {
            "frame": {
                "origin_model_world": vector_to_dict(body.frame.origin_model_world),
                "right_axis_model_world": vector_to_dict(body.frame.right_axis_model_world),
                "up_axis_model_world": vector_to_dict(body.frame.up_axis_model_world),
                "normal_axis_model_world": vector_to_dict(body.frame.normal_axis_model_world)
            },
            "shoulder_width_model_world": body.shoulder_width_model_world,
            "torso_height_model_world": body.torso_height_model_world,
            "left_arm_length_model_world": body.left_arm_length_model_world,
            "right_arm_length_model_world": body.right_arm_length_model_world,
            "left_upper_arm_direction_body": vector_to_dict(body.left_upper_arm_direction_body),
            "left_forearm_direction_body": vector_to_dict(body.left_forearm_direction_body),
            "right_upper_arm_direction_body": vector_to_dict(body.right_upper_arm_direction_body),
            "right_forearm_direction_body": vector_to_dict(body.right_forearm_direction_body),
            "left_wrist_displacement_normalized_body": vector_to_dict(body.left_wrist_displacement_normalized_body),
            "right_wrist_displacement_normalized_body": vector_to_dict(body.right_wrist_displacement_normalized_body)
        }

    if derived.left_hand is not None:
        left = derived.left_hand

        result["left_hand"] = {
            "palm_width_model_world": left.palm_width_model_world,
            "palm_length_model_world": left.palm_length_model_world,
            "pinch_ratio": left.pinch_ratio,
            "index_direction_model_world": vector_to_dict(left.index_direction_model_world),
            "palm_normal_model_world": vector_to_dict(left.palm_normal_model_world)
        }

    if derived.right_hand is not None:
        right = derived.right_hand

        result["right_hand"] = {
            "palm_width_model_world": right.palm_width_model_world,
            "palm_length_model_world": right.palm_length_model_world,
            "pinch_ratio": right.pinch_ratio,
            "index_direction_model_world": vector_to_dict(right.index_direction_model_world),
            "palm_normal_model_world": vector_to_dict(right.palm_normal_model_world)
        }

    return result

def file_identity(path: Path) -> dict:
    resolved = path.expanduser().resolve()

    if not resolved.is_file():
        raise FileNotFoundError(f"Model asset not found: {resolved}")

    digest = hashlib.sha256()

    with resolved.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)

    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": digest.hexdigest()
    }

def retargeting_result_to_dict(result):
    if result is None:
        return None

    return {
        "frame_sequence_id": result.frame_sequence_id,
        "measurement_timestamp_s": result.measurement_timestamp_s,
        "active": result.active,
        "valid": result.valid,
        "target_pose":
        (
            None if result.target_pose is None
            else {
                "position": vector_to_dict(result.target_pose.position),
                "orientation_xyzw": {
                    "x": result.target_pose.orientation_xyzw.x,
                    "y": result.target_pose.orientation_xyzw.y,
                    "z": result.target_pose.orientation_xyzw.z,
                    "w": result.target_pose.orientation_xyzw.w
                }
            }
        ),
        "operator_delta_normalized_body": vector_to_dict(result.operator_delta_normalized_body),
        "deadbanded_delta_normalized_body": vector_to_dict(result.deadbanded_delta_normalized_body),
        "mapped_delta_robot": vector_to_dict(result.mapped_delta_robot),
        "workspace_limited": result.workspace_limited,
        "speed_limited": result.speed_limited,
        "reasons": list(result.reasons)
    }

def gripper_aperture_result_to_dict(result: GripperApertureResult | None):
    if result is None:
        return None

    return {
        "frame_sequence_id": result.frame_sequence_id,
        "measurement_timestamp_s": result.measurement_timestamp_s,
        "hand_side": result.hand_side.value,
        "measurement_valid": result.measurement_valid,
        "normalized_aperture": result.normalized_aperture,
        "fully_open": result.fully_open,
        "fully_closed": result.fully_closed,
        "pinch_ratio": result.pinch_ratio,
        "reasons": list(result.reasons)
    }

def supervisor_result_to_dict(result: CommandSupervisorResult | None):
    if result is None:
        return None

    return {
        "cycle_sequence_id": result.cycle_sequence_id,
        "decision_timestamp_s": result.decision_timestamp_s,
        "previous_state": result.previous_state.value,
        "state": result.state.value,
        "state_changed": result.state_changed,
        "motion_permitted": result.motion_permitted,
        "permitted_target":
        (
            None if result.permitted_target is None
            else
            {
                "position": vector_to_dict(result.permitted_target.position),
                "orientation_xyzw":
                {
                    "x": result.permitted_target.orientation_xyzw.x,
                    "y": result.permitted_target.orientation_xyzw.y,
                    "z": result.permitted_target.orientation_xyzw.z,
                    "w": result.permitted_target.orientation_xyzw.w
                }
            }
        ),
        "gripper_command_permitted": result.gripper_command_permitted,
        "permitted_gripper_aperture": result.permitted_gripper_aperture,
        "consecutive_valid_cycles": result.consecutive_valid_cycles,
        "reasons": list(result.reasons),
        "gripper_reasons": list(result.gripper_reasons)
    }

def selected_derived_hand(derived, hand_side: str):
    if hand_side == "left":
        return derived.left_hand

    if hand_side == "right":
        return derived.right_hand

    raise ValueError(f"Unsupported gesture hand: {hand_side}")

def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None

    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1.")

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = probability * (len(ordered) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))

    if lower_index == upper_index:
        return ordered[lower_index]

    fraction = position - lower_index

    return (ordered[lower_index] * (1.0 - fraction) + ordered[upper_index] * fraction)

def sample_summary(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None

    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "std_population":
        (
            statistics.pstdev(values)
            if len(values) > 1
            else 0.0
        ),
        "minimum": min(values),
        "p05": percentile(values, 0.05),
        "p25": percentile(values, 0.25),
        "median": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "p95": percentile(values, 0.95),
        "maximum": max(values)
    }

def print_sample_summary(label: str, values: list[float], invalid_count: int) -> None:
    summary = sample_summary(values)

    print(f"{label} valid samples: {len(values)}")
    print(f"{label} labeled frames without valid H6 hand geometry: {invalid_count}")

    if summary is None:
        print(f"{label} pinch statistics: unavailable")
        return

    print(
        f"{label} pinch mean/std: "
        f"{summary['mean']:.3f} / {summary['std_population']:.3f}"
    )
    print(
        f"{label} pinch min/p05/p25/median/p75/p95/max: "
        f"{summary['minimum']:.3f} / {summary['p05']:.3f} / "
        f"{summary['p25']:.3f} / {summary['median']:.3f} / "
        f"{summary['p75']:.3f} / {summary['p95']:.3f} / "
        f"{summary['maximum']:.3f}"
    )

class JsonlDiagnosticLogger:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir.expanduser().resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        self.path = self.log_dir / ("human_tracking_"f"{timestamp_utc}.jsonl")
        self._file = self.path.open("x", encoding="utf-8", newline="\n")
        self.record_count = 0

    def write(self, record_type: str, payload: dict) -> None:
        record = {
            "schema_name": LOG_SCHEMA_NAME,
            "schema_version": LOG_SCHEMA_VERSION,
            "record_type": record_type,
            **payload
        }

        self._file.write(json.dumps(json_safe(record), separators=(",", ":"), ensure_ascii=False) + "\n")
        self._file.flush()
        self.record_count += 1

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

@dataclass(frozen=True)
class _DiagnosticConfiguration:
    selected_control_arm: ControlArm
    validation: ObservationValidationConfig
    temporal: TemporalTrackingConfig
    smoothing: LandmarkSmoothingConfig
    derived: DerivedGeometryConfig
    control_arm: ArmSide
    gesture_hand: str
    p4e_enabled: bool
    aperture: GripperApertureConfig | None
    gripper_hand_side: HandSide | None
    supervisor: CommandSupervisorConfig | None
    h8: RetargetingConfig
    h8_virtual_reference_pose: RobotAgnosticPose

@dataclass
class _DiagnosticRuntime:
    acquisition: CameraAcquisition
    backend: MediaPipeTrackingBackend
    adapter: MediaPipeObservationAdapter
    tracker: HumanTemporalTracker
    smoother: HumanLandmarkSmoother
    retargeter: RelativeRetargeter
    aperture_tracker: GripperApertureTracker | None
    supervisor: CommandSupervisor | None

@dataclass
class _PinchCalibrationState:
    label: str | None = None
    open_samples: list[float] = field(default_factory=list)
    closed_samples: list[float] = field(default_factory=list)
    open_invalid_frames: int = 0
    closed_invalid_frames: int = 0
    label_change_count: int = 0

    def reset(self) -> None:
        self.label = None
        self.open_samples.clear()
        self.closed_samples.clear()
        self.open_invalid_frames = 0
        self.closed_invalid_frames = 0
        self.label_change_count = 0

    def record_frame(self, hand_side: str, measurement_started: bool, pinch_ratio: float | None) -> dict:
        result = {
            "hand_side": hand_side,
            "label": self.label,
            "sample_valid": False,
            "pinch_ratio": None,
        }
        if not measurement_started or self.label is None:
            return result

        valid = pinch_ratio is not None and math.isfinite(pinch_ratio) and pinch_ratio >= 0.0

        if valid:
            result["sample_valid"] = True
            result["pinch_ratio"] = pinch_ratio
            if self.label == "open":
                self.open_samples.append(pinch_ratio)
            elif self.label == "closed":
                self.closed_samples.append(pinch_ratio)
            else:
                raise RuntimeError("Unknown pinch calibration label.")
        elif self.label == "open":
            self.open_invalid_frames += 1
        elif self.label == "closed":
            self.closed_invalid_frames += 1
        else:
            raise RuntimeError("Unknown pinch calibration label.")
        return result

def _handle_pinch_label_key(
    key: int,
    measurement_started: bool,
    calibration: _PinchCalibrationState,
    logger: JsonlDiagnosticLogger | None,
    derived,
    hand_side: str,
) -> bool:
    if not measurement_started or key == 0xFF:
        return False
    key_char = chr(key).lower()
    if key_char not in {"o", "c", "x"}:
        return False

    previous_label = calibration.label
    calibration.label = {"o": "open", "c": "closed", "x": None}[key_char]
    calibration.label_change_count += 1

    if logger is not None:
        if key_char == "x":
            payload = {
                "label": None,
                "previous_label": previous_label,
                "hand_side": hand_side,
                "frame_sequence_id": derived.frame_sequence_id,
                "measurement_timestamp_s": derived.measurement_timestamp_s,
            }

        else:
            payload = {
                "label": calibration.label,
                "hand_side": hand_side,
                "frame_sequence_id": derived.frame_sequence_id,
                "measurement_timestamp_s": derived.measurement_timestamp_s,
            }
        logger.write("pinch_label_change", payload)

    if key_char == "o":
        print("P3 label = OPEN. Hold a clearly open thumb-index span. Press X before changing pose.")
    elif key_char == "c":
        print("P3 label = CLOSED/PINCH. Hold a comfortable thumb-index pinch. Press X before changing pose.")
    else:
        print("P3 labeling stopped. Change hand pose before starting the next labeled segment.")
    return True

@dataclass
class _SupervisorRequestFlags:
    activation: bool = False
    enable: bool = False
    disable: bool = False

    def clear(self) -> None:
        self.activation = False
        self.enable = False
        self.disable = False

@dataclass
class _P4EAutoState:
    launch_s: float
    activation_sent: bool = False
    active_started_s: float | None = None
    disable_sent: bool = False
    end_reason: str | None = None

    def reset_for_measurement(self) -> None:
        self.activation_sent = False
        self.active_started_s = None
        self.disable_sent = False
        self.end_reason = None

@dataclass(frozen=True)
class _P4EFrameDecision:
    aperture: GripperApertureResult
    supervisor: CommandSupervisorResult
    decision_age_s: float

def _run_p4e_frame(
    aperture_tracker: GripperApertureTracker,
    supervisor: CommandSupervisor,
    derived,
    tracking,
    retargeting_result,
    requests: _SupervisorRequestFlags,
) -> _P4EFrameDecision:
    aperture = aperture_tracker.update(derived)
    decision_timestamp_s = time.perf_counter()
    decision_age_s = decision_timestamp_s - derived.measurement_timestamp_s

    supervisor_result = supervisor.update(
        SupervisorCycleInput(
            cycle_sequence_id = derived.frame_sequence_id,
            decision_timestamp_s = decision_timestamp_s,
            human_valid = tracking.body.valid_for_control and derived.body is not None,
            human_frame_sequence_id = derived.frame_sequence_id,
            human_measurement_timestamp_s = derived.measurement_timestamp_s,
            robot_state_valid = True,
            robot_state_timestamp_s = derived.measurement_timestamp_s,
            transform_available = True,
            transform_timestamp_s = None,
            retargeting_result = retargeting_result,
            controller_ready = True,
            gripper_aperture_result = aperture,
            activation_requested = requests.activation,
            enable_requested = requests.enable,
            disable_requested = requests.disable
        )
    )

    requests.clear()
    return _P4EFrameDecision(aperture=aperture, supervisor=supervisor_result, decision_age_s=decision_age_s)

def _p4e_auto_status(
    args: argparse.Namespace,
    auto: _P4EAutoState,
    measurement_started: bool,
    retargeting_activation_count: int,
    supervisor_result: CommandSupervisorResult | None,
) -> str:
    now_s = time.perf_counter()
    if not measurement_started:
        remaining_s = max(0.0, args.p4e_auto_start_delay_s - (now_s - auto.launch_s))
        return f"P4E AUTO: position yourself; start in {remaining_s:.1f}s"
    if retargeting_activation_count == 0:
        return ("P4E AUTO: waiting for valid body + aperture to capture reference")
    if not auto.activation_sent:
        return "P4E AUTO: reference captured; preparing activation"
    if supervisor_result is not None and supervisor_result.state == SupervisorState.ACQUIRING:
        return "P4E AUTO: ACQUIRING stable inputs"
    if supervisor_result is not None and supervisor_result.state == SupervisorState.READY and auto.active_started_s is None:
        return "P4E AUTO: READY; scripted enable pending"
    if supervisor_result is not None and supervisor_result.state == SupervisorState.ACTIVE and auto.active_started_s is not None:
        remaining_s = max(0.0, args.p4e_auto_active_duration_s - (now_s - auto.active_started_s))
        return ("P4E AUTO: ACTIVE; move arm/aperture "f"({remaining_s:.1f}s remaining)")
    if supervisor_result is not None and supervisor_result.state == SupervisorState.HOLD:
        if auto.active_started_s is None:
            return "P4E AUTO: HOLD before first ACTIVE; waiting fail-closed"
        remaining_s = max(0.0, args.p4e_auto_active_duration_s - (now_s - auto.active_started_s))
        return ("P4E AUTO: HOLD; commands suppressed, no auto re-enable "f"({remaining_s:.1f}s remaining; Q stops)")
    if auto.disable_sent:
        return "P4E AUTO: disabling / finishing"
    return "P4E AUTO: running"

def _apply_p4e_auto_key(
    args: argparse.Namespace,
    auto: _P4EAutoState,
    key: int,
    measurement_started: bool,
    retargeting_activation_count: int,
    tracking,
    derived,
    gripper_aperture_result: GripperApertureResult | None,
    retargeting_result,
    supervisor_result: CommandSupervisorResult | None,
) -> int:
    if not args.p4e_auto_sequence or key != 0xFF:
        return key

    now_s = time.perf_counter()
    auto_key = None

    if not measurement_started:
        if now_s - auto.launch_s >= args.p4e_auto_start_delay_s:
            auto_key = ord("S")
    elif retargeting_activation_count == 0:
        if (
            tracking.body.valid_for_control and derived.body is not None
            and gripper_aperture_result is not None and gripper_aperture_result.measurement_valid
        ):
            auto_key = ord("R")
    elif not auto.activation_sent:
        if retargeting_result is not None and retargeting_result.active and retargeting_result.valid:
            auto_key = ord("T")
            auto.activation_sent = True
    elif supervisor_result is not None and supervisor_result.state == SupervisorState.READY and auto.active_started_s is None:
        auto_key = ord("E")
    elif auto.disable_sent and supervisor_result is not None and supervisor_result.state == SupervisorState.DISABLED:
        auto_key = ord("Q")
    elif auto.active_started_s is not None:
        if now_s - auto.active_started_s >= args.p4e_auto_active_duration_s and not auto.disable_sent:
            auto_key = ord("D")
            auto.disable_sent = True
            auto.end_reason = "diagnostic_window_elapsed"
        elif supervisor_result is not None and supervisor_result.state == SupervisorState.HOLD:
            pass
        elif supervisor_result is not None and supervisor_result.state == SupervisorState.ACTIVE:
            pass
    elif supervisor_result is not None and supervisor_result.state == SupervisorState.ACTIVE:
        auto.active_started_s = now_s
        print("P4E AUTO: supervisor ACTIVE. Move the selected arm and vary thumb-index aperture while remaining in the tracking "
            "volume. The diagnostic window will remain open for " f"{args.p4e_auto_active_duration_s:.1f} s unless Q is pressed."
        )

    return auto_key if auto_key is not None else key

def _handle_supervisor_request_key(
    key: int,
    measurement_started: bool,
    p4e_enabled: bool,
    requests: _SupervisorRequestFlags,
    logger: JsonlDiagnosticLogger | None,
    derived,
) -> bool:
    request_specs = {
        ord("t"): (
            "activation",
            "activation",
            "P4E teleoperation request ignored: integrated aperture/supervisor is not configured.",
            "P4E teleoperation requested. Supervisor will enter ACQUIRING on the next measured frame.",
        ),
        ord("e"): (
            "enable",
            "enable_or_reenable",
            "P4E enable request ignored: integrated aperture/supervisor is not configured.",
            "P4E enable/re-enable requested for the next measured frame.",
        ),
        ord("d"): (
            "disable",
            "disable",
            "P4E disable request ignored: integrated aperture/supervisor is not configured.",
            "P4E disable requested for the next measured frame.",
        ),
    }
    spec = request_specs.get(ord(chr(key).lower())) if key != 0xFF else None
    if not measurement_started or spec is None:
        return False

    field_name, request_name, disabled_message, accepted_message = spec
    if not p4e_enabled:
        print(disabled_message)
        return True

    setattr(requests, field_name, True)
    if logger is not None:
        logger.write("supervisor_request",
            {
                "request": request_name,
                "frame_sequence_id": derived.frame_sequence_id,
                "measurement_timestamp_s": derived.measurement_timestamp_s,
            }
        )
    print(accepted_message)
    return True

def _build_diagnostic_configuration(args: argparse.Namespace) -> _DiagnosticConfiguration:
    selected_control_arm = ControlArm(args.control_arm)
    validation = ObservationValidationConfig(
        selected_control_arm = selected_control_arm,
        min_body_visibility = 0.50,
        min_body_presence = 0.50,
        min_handedness_score = 0.50,
        enable_pose_hand_association = (args.hand_association == "pose_wrist"),
        max_hand_wrist_pose_distance_shoulder_widths = (args.hand_association_max_distance_shoulder_widths)
    )
    temporal = TemporalTrackingConfig(consecutive_valid_required=3, dropout_timeout_s=0.25)
    smoothing = LandmarkSmoothingConfig(time_constant_s=args.smoothing_time_constant)
    derived = DerivedGeometryConfig(
        selected_control_arm = selected_control_arm,
        min_length_model_world = 1e-5,
        min_palm_width_to_palm_length_ratio = (args.hand_min_palm_width_to_palm_length_ratio)
    )
    control_arm = ArmSide(args.control_arm)
    gesture_hand = args.gesture_hand or args.control_arm

    aperture_reference_count = sum(value is not None for value in (args.closed_reference_ratio, args.open_reference_ratio))
    if aperture_reference_count == 1:
        raise ValueError("P4E requires both --closed-reference-ratio and --open-reference-ratio, or neither.")

    p4e_enabled = aperture_reference_count == 2
    if p4e_enabled and args.gripper_loss_policy is None:
        raise ValueError("P4E requires an explicit --gripper-loss-policy when continuous-aperture references are supplied.")
    if not p4e_enabled and args.gripper_loss_policy is not None:
        raise ValueError("--gripper-loss-policy requires explicit aperture references.")
    if args.p4e_auto_sequence and not p4e_enabled:
        raise ValueError("--p4e-auto-sequence requires explicit aperture references and --gripper-loss-policy.")
    if (not math.isfinite(args.p4e_auto_start_delay_s) or args.p4e_auto_start_delay_s < 0.0 ):
        raise ValueError( "--p4e-auto-start-delay-s must be finite and non-negative.")
    if (not math.isfinite(args.p4e_auto_active_duration_s) or args.p4e_auto_active_duration_s <= 0.0):
        raise ValueError("--p4e-auto-active-duration-s must be finite and positive.")

    aperture = None
    gripper_hand_side = None
    supervisor = None
    if p4e_enabled:
        aperture = GripperApertureConfig(closed_reference_ratio=args.closed_reference_ratio, open_reference_ratio=args.open_reference_ratio)
        gripper_hand_side = HandSide(gesture_hand)
        supervisor = CommandSupervisorConfig(
            consecutive_valid_required = 3,
            max_human_age_s = 0.20,
            max_robot_state_age_s = 0.20,
            max_dynamic_transform_age_s = 0.20,
            max_target_age_s = 0.20,
            max_human_robot_skew_s = 0.05,
            max_human_transform_skew_s = 0.05,
            max_human_target_skew_s = 0.01,
            gripper_loss_policy = GripperLossPolicy(args.gripper_loss_policy)
        )

    h8 = RetargetingConfig(
        axis_mapping = AxisMapping.identity(),
        scale_robot_per_normalized_body = Vector3(x = 1.0, y = 1.0, z = 1.0),
        deadband_normalized_body=Vector3(x = 0.0, y = 0.0, z = 0.0),
        workspace_bounds = CartesianBounds(
            minimum = Vector3(x = -100.0, y = -100.0, z = -100.0),
            maximum = Vector3(x = 100.0, y = 100.0, z = 100.0)
        ),
        max_cartesian_speed_robot_per_s = None
    )
    reference_pose = RobotAgnosticPose(
        position = Vector3(x = 0.0, y = 0.0, z = 0.0),
        orientation_xyzw=Quaternion(x = 0.0, y = 0.0, z = 0.0, w = 1.0),
    )
    return _DiagnosticConfiguration(
        selected_control_arm = selected_control_arm,
        validation = validation,
        temporal = temporal,
        smoothing = smoothing,
        derived = derived,
        control_arm = control_arm,
        gesture_hand = gesture_hand,
        p4e_enabled = p4e_enabled,
        aperture = aperture,
        gripper_hand_side = gripper_hand_side,
        supervisor = supervisor,
        h8 = h8,
        h8_virtual_reference_pose = reference_pose,
    )

def _print_diagnostic_configuration(args: argparse.Namespace, config: _DiagnosticConfiguration) -> None:
    print("=" * 60)
    print("H2/H4/H5/H6/H7/H8/P3/P4E - Live Human Tracking + Integrated Pre-ROS Diagnostic (schema v9)")
    print("=" * 60)
    print(f"Camera index: {args.camera}")
    print(f"Pose model:   {args.pose_model}")
    print(f"Hand model:   {args.hand_model}")
    print()
    print("HT2 MediaPipe hand-landmarker thresholds:")
    print("  Minimum detection confidence: " f"{args.hand_min_detection_confidence:.2f}")
    print("  Minimum presence confidence:  "f"{args.hand_min_presence_confidence:.2f}")
    print("  Minimum tracking confidence:  "f"{args.hand_min_tracking_confidence:.2f}")
    print("  These are perception tuning values only; they are not robot safety limits.")
    print()
    print("Current diagnostic thresholds:")
    print(f"  Minimum body visibility:  {config.validation.min_body_visibility:.2f}")
    print(f"  Minimum body presence:    {config.validation.min_body_presence:.2f}")
    print(f"  Minimum handedness score: {config.validation.min_handedness_score:.2f}")
    print()
    print("HT1 hand-association settings:")
    print(f"  Mode: {args.hand_association}")
    if config.validation.enable_pose_hand_association:
        print("  Maximum hand/pose-wrist distance: "
            f"{config.validation.max_hand_wrist_pose_distance_shoulder_widths:.2f} "
            "shoulder widths")
        print("  Gate type: provisional image-space value derived from recorded diagnostics; not a physical robot limit")
    print()
    print("Temporal diagnostic settings:")
    print("  Consecutive valid frames required: "f"{config.temporal.consecutive_valid_required}")
    print(f"  Dropout timeout: {config.temporal.dropout_timeout_s:.2f} s")
    print()
    print("H5 smoothing settings:")
    print(f"  Time constant: {config.smoothing.time_constant_s:.3f} s")
    print()
    print("H6 derived-geometry settings:")
    print("  Minimum non-degenerate model-world length: "f"{config.derived.min_length_model_world:.1e}")
    print("  HT4 pinch normalization: thumb-index tip distance / wrist-to-middle-MCP palm length")
    if config.derived.min_palm_width_to_palm_length_ratio is None:
        print("  HT3 palm-shape plausibility gate: DISABLED")
    else:
        print("  HT3 minimum palm-width/palm-length ratio: "f"{config.derived.min_palm_width_to_palm_length_ratio:.2f}")
        print("  HT3 ratio is a provisional perception-quality value derived from recorded hand geometry; not a physical robot limit")
    print()
    print("H8 virtual-retargeting settings:")
    print(f"  Control arm: {config.control_arm.value}")
    print("  Axis mapping: identity")
    print("  Scale: 1 virtual target unit per normalized body unit")
    print("  Deadband: 0 for this diagnostic")
    print("  Physical workspace/speed limits: NOT selected")
    print()
    print("P3 pinch-calibration settings:")
    print(f"  Gesture hand: {config.gesture_hand}")
    print("  OPEN/CLOSED thresholds: NOT selected")
    print("  Minimum valid samples per label for this diagnostic: 30")
    print()
    print("P4E integrated pre-ROS settings:")
    if not config.p4e_enabled:
        print("  Integrated aperture/supervisor: DISABLED (no aperture references supplied)")
    else:
        assert config.aperture is not None
        assert config.gripper_hand_side is not None
        assert config.supervisor is not None
        print("  Integrated aperture/supervisor: ENABLED")
        print(f"  Aperture hand: {config.gripper_hand_side.value}")
        print("  Closed/open reference ratios: "
            f"{config.aperture.closed_reference_ratio:.3f} / "
            f"{config.aperture.open_reference_ratio:.3f}")
        print("  Gripper-loss policy: " f"{config.supervisor.gripper_loss_policy.value}")
        print("  Robot state / transform / controller inputs: SYNTHETIC diagnostic inputs only")
        print("  Supervisor timing limits: P1 SYNTHETIC test values; not Franka limits")

        if args.p4e_auto_sequence:
            print("  Control mode: HANDS-FREE scripted diagnostic")
            print(f"  Automatic start delay: {args.p4e_auto_start_delay_s:.1f} s (operator-interface timing only)")
            print("  Post-ACTIVE diagnostic window: " f"{args.p4e_auto_active_duration_s:.1f} s (diagnostic timing only)")
            print("  Automatic re-enable after HOLD: DISABLED (explicit recovery semantics preserved)")
        else:
            print("  Control mode: manual camera-window keys")
    print()
    print("These are preliminary diagnostic thresholds, not final thesis values.")
    if config.validation.enable_pose_hand_association:
        print("Anatomical Left/Right hand buckets use current-frame pose-wrist association; raw MediaPipe handedness is retained as metadata.")
    else:
        print("Left/Right uses the raw MediaPipe handedness classification.")
    print()

def _open_diagnostic_runtime(args: argparse.Namespace, config: _DiagnosticConfiguration) -> _DiagnosticRuntime:
    try:
        acquisition = CameraAcquisition(camera_index=args.camera)
        acquisition.open()
    except Exception as exc:
        print("RESULT: FAIL")
        print(f"Camera error: {type(exc).__name__}: {exc}")
        raise SystemExit(1)

    try:
        backend = MediaPipeTrackingBackend(
            pose_model_path = args.pose_model,
            hand_model_path = args.hand_model,
            min_hand_detection_confidence = (args.hand_min_detection_confidence),
            min_hand_presence_confidence = (args.hand_min_presence_confidence),
            min_hand_tracking_confidence = (args.hand_min_tracking_confidence)
        )
    except Exception as exc:
        acquisition.close()
        print("RESULT: FAIL")
        print(f"Backend error: {type(exc).__name__}: {exc}")
        raise SystemExit(1)

    return _DiagnosticRuntime(
        acquisition = acquisition,
        backend = backend,
        adapter = MediaPipeObservationAdapter(),
        tracker = HumanTemporalTracker(config.temporal),
        smoother = HumanLandmarkSmoother(config.smoothing),
        retargeter = RelativeRetargeter(
            arm_side = config.control_arm,
            config = config.h8),
        aperture_tracker = (None if config.aperture is None
            else GripperApertureTracker(
                hand_side = config.gripper_hand_side,
                config = config.aperture,
            )),
        supervisor = (None if config.supervisor is None
            else CommandSupervisor(config.supervisor))
    )

def _print_run_instructions(args: argparse.Namespace, p4e_enabled: bool) -> None:
    print("Models loaded successfully.")
    print()
    messages = (
        "Preview starts immediately, but statistics are NOT collected yet.",
        "Stand up, move into the intended test position, and make sure the required body landmarks are visible.",
        "When you are ready, press S in the camera window to start the measurement.",
        "After S, hold still briefly, then move your arms slowly through several comfortable positions.",
        "Keep the selected gesture hand visible and avoid rapid motion while collecting labeled samples.",
        "Raw pose is green, raw hands are yellow, and smoothed control-valid landmarks are magenta.",
        "H6 reports body-relative wrist displacement and palm-length-normalized pinch ratios; it does not issue gestures or robot commands.",
        "H7 creates one JSONL log when S is pressed. Preview frames before S are not logged.",
        "H8 virtual retargeting remains available for regression. Press R only if you also want to exercise it; R is not required for P3.",
        "H8 uses virtual unitless target coordinates only. No robot, physical workspace, or physical speed limit is being assumed.",
    )
    for message in messages:
        print(message)

    if p4e_enabled:
        print()
        if args.p4e_auto_sequence:
            print("P4E hands-free sequence is ENABLED. After launch, move into the intended camera position and remain there.")
            print(f"Measurement starts automatically after {args.p4e_auto_start_delay_s:.1f} s. H8 reference capture then waits"
                " for current valid body and aperture data; no keyboard approach is required.")
            print("The scripted diagnostic requests teleoperation, waits for READY, issues the initial enable,"
                " then keeps the live diagnostic open for " f"{args.p4e_auto_active_duration_s:.1f} s after first ACTIVE before "
                "disabling and quitting automatically.")
            print("If ACTIVE enters HOLD, commands remain suppressed and the diagnostic keeps running;"
                " it never automatically re-enables after recovery. Press Q in the camera window to stop early.")
        else:
            print("P4E controls after S: R captures the H8 reference, T requests teleoperation,"
                " E enables/re-enables after READY/recovery, and D disables teleoperation.")
        print("P4E uses synthetic robot-state, static-transform, and controller-ready inputs."
            "It does not represent a connected Franka robot.")

    print()
    for message in (
        "P3 pinch calibration uses the selected gesture hand only.",
        "After S: press O while holding a clearly OPEN hand; press X to stop labeling.",
        "Then form a comfortable thumb-index pinch, press C to label CLOSED/pinch samples, and press X to stop.",
        "Repeat OPEN and CLOSED segments several times. Only frames with valid H6 hand geometry become samples.",
        "No OPEN/CLOSED threshold is selected automatically.",
        "Press Q to stop the test."):
        print(message)
    print("-" * 60)

@dataclass(frozen=True)
class _RunReportData:
    args: argparse.Namespace
    logger: JsonlDiagnosticLogger | None
    log_path: Path | None
    gesture_hand: str
    calibration: _PinchCalibrationState
    p4e_auto: _P4EAutoState
    p4e_enabled: bool
    measurement_started: bool
    duration_s: float
    average_processing_ms: float
    average_loop_fps: float
    frame_count: int
    pose_frame_count: int
    hand_frame_count: int
    total_hand_detections: int
    max_hands_seen: int
    two_hand_raw_frame_count: int
    left_raw_detected_count: int
    right_raw_detected_count: int
    left_pose_associated_count: int
    right_pose_associated_count: int
    left_pose_override_count: int
    right_pose_override_count: int
    visualization_frame_count: int
    body_valid_count: int
    left_hand_valid_count: int
    right_hand_valid_count: int
    both_hands_valid_count: int
    body_temporal_valid_count: int
    left_temporal_valid_count: int
    right_temporal_valid_count: int
    both_hands_temporal_valid_count: int
    body_state_counts: Counter
    left_state_counts: Counter
    right_state_counts: Counter
    body_rejection_reasons: Counter
    left_hand_rejection_reasons: Counter
    right_hand_rejection_reasons: Counter
    smoothed_body_frame_count: int
    smoothed_left_frame_count: int
    smoothed_right_frame_count: int
    body_separation_sum_px: float
    body_separation_samples: int
    left_separation_sum_px: float
    left_separation_samples: int
    right_separation_sum_px: float
    right_separation_samples: int
    derived_body_frame_count: int
    derived_left_frame_count: int
    derived_right_frame_count: int
    left_wrist_norm_sum: float
    left_wrist_norm_samples: int
    right_wrist_norm_sum: float
    right_wrist_norm_samples: int
    left_pinch_ratio_sum: float
    left_pinch_ratio_samples: int
    left_pinch_ratio_min: float
    left_pinch_ratio_max: float
    right_pinch_ratio_sum: float
    right_pinch_ratio_samples: int
    right_pinch_ratio_min: float
    right_pinch_ratio_max: float
    body_derived_reasons: Counter
    left_derived_reasons: Counter
    right_derived_reasons: Counter
    retargeting_activation_count: int
    retargeting_active_frame_count: int
    retargeting_valid_target_count: int
    retargeting_invalid_target_count: int
    retargeting_workspace_limited_count: int
    retargeting_speed_limited_count: int
    max_operator_delta_norm: float
    max_virtual_target_norm: float
    p4e_supervisor_cycle_count: int
    p4e_aperture_valid_count: int
    p4e_supervisor_state_counts: Counter
    p4e_motion_permitted_count: int
    p4e_gripper_permitted_count: int
    p4e_permission_violation_count: int

def _print_count_with_percentage(label: str, count: int, frame_count: int) -> None:
    print(f"{label}: {count} " f"({percentage(count, frame_count):.1f}%)")

def _print_reason_summary(title: str, reasons: Counter, frame_count: int) -> None:
    print()
    print(title)
    if not reasons:
        print("  none")
        return
    for reason, count in reasons.most_common():
        print(
            f"  {reason}: {count} "
            f"({percentage(count, frame_count):.1f}% of frames)")

def _finalize_run(report: _RunReportData) -> None:
    print()
    print("-" * 60)

    if not report.measurement_started or report.frame_count == 0:
        print("No measurement was recorded.")
        print("Run the test again and press S after you are in the intended test position.")
        raise SystemExit(0)

    print(f"Frames processed: {report.frame_count}")
    _print_count_with_percentage(
        "Frames with raw pose",
        report.pose_frame_count,
        report.frame_count)

    _print_count_with_percentage(
        "Frames with >=1 raw hand",
        report.hand_frame_count,
        report.frame_count)

    print(f"Total raw hand detections: {report.total_hand_detections}")
    print(f"Maximum simultaneous raw hands: {report.max_hands_seen}")
    _print_count_with_percentage("Frames with 2 raw hands",
        report.two_hand_raw_frame_count,
        report.frame_count)

    _print_count_with_percentage(
        "Frames with raw Left hand",
        report.left_raw_detected_count,
        report.frame_count)

    _print_count_with_percentage(
        "Frames with raw Right hand",
        report.right_raw_detected_count,
        report.frame_count)

    print()
    print("HT1 hand-association summary:")
    print(f"Mode: {report.args.hand_association}")
    if report.args.hand_association == "pose_wrist":
        _print_count_with_percentage(
            "Frames with pose-associated Left hand",
            report.left_pose_associated_count,
            report.frame_count,
        )
        _print_count_with_percentage(
            "Frames with pose-associated Right hand",
            report.right_pose_associated_count,
            report.frame_count,
        )
        print(
            "Pose-associated Left hands overriding backend side label: "
            f"{report.left_pose_override_count}")
        print(
            "Pose-associated Right hands overriding backend side label: "
            f"{report.right_pose_override_count}")

    print()
    _print_count_with_percentage(
        "Available for visualization",
        report.visualization_frame_count,
        report.frame_count)
    _print_count_with_percentage(
        "Body per-frame control-valid",
        report.body_valid_count,
        report.frame_count)
    _print_count_with_percentage(
        "Left-hand per-frame control-valid",
        report.left_hand_valid_count,
        report.frame_count)
    _print_count_with_percentage(
        "Right-hand per-frame control-valid",
        report.right_hand_valid_count,
        report.frame_count)
    _print_count_with_percentage(
        "Both hands per-frame control-valid",
        report.both_hands_valid_count,
        report.frame_count)
    print()
    print("After temporal gating:")
    _print_count_with_percentage(
        "Body temporal control-valid",
        report.body_temporal_valid_count,
        report.frame_count,
    )
    _print_count_with_percentage(
        "Left temporal control-valid",
        report.left_temporal_valid_count,
        report.frame_count,
    )
    _print_count_with_percentage(
        "Right temporal control-valid",
        report.right_temporal_valid_count,
        report.frame_count,
    )
    _print_count_with_percentage(
        "Both hands temporal control-valid",
        report.both_hands_temporal_valid_count,
        report.frame_count,
    )

    print()
    print(f"Body temporal states: {format_state_counts(report.body_state_counts)}")
    print(f"Left temporal states: {format_state_counts(report.left_state_counts)}")
    print(f"Right temporal states: {format_state_counts(report.right_state_counts)}")

    _print_reason_summary(
        "Body rejection reasons:",
        report.body_rejection_reasons,
        report.frame_count,
    )
    _print_reason_summary(
        "Left-hand rejection reasons:",
        report.left_hand_rejection_reasons,
        report.frame_count,
    )
    _print_reason_summary(
        "Right-hand rejection reasons:",
        report.right_hand_rejection_reasons,
        report.frame_count,
    )

    print()
    print("H5 smoothing summary:")
    _print_count_with_percentage(
        "Smoothed body frames",
        report.smoothed_body_frame_count,
        report.frame_count,
    )
    _print_count_with_percentage(
        "Smoothed Left-hand frames",
        report.smoothed_left_frame_count,
        report.frame_count,
    )
    _print_count_with_percentage(
        "Smoothed Right-hand frames",
        report.smoothed_right_frame_count,
        report.frame_count,
    )

    mean_body_separation_px = (
        report.body_separation_sum_px / report.body_separation_samples
        if report.body_separation_samples > 0 else 0.0)
    mean_left_separation_px = (
        report.left_separation_sum_px / report.left_separation_samples
        if report.left_separation_samples > 0 else 0.0)
    mean_right_separation_px = (
        report.right_separation_sum_px / report.right_separation_samples
        if report.right_separation_samples > 0 else 0.0)
    print(f"Mean raw-to-smoothed body separation: {mean_body_separation_px:.2f} px")
    print(f"Mean raw-to-smoothed Left-hand separation: {mean_left_separation_px:.2f} px")
    print(f"Mean raw-to-smoothed Right-hand separation: {mean_right_separation_px:.2f} px")
    print("The separation values describe smoothing deviation; they are not direct latency measurements.")
    print()
    print("H6 derived-geometry summary:")
    _print_count_with_percentage(
        "Derived body frames",
        report.derived_body_frame_count,
        report.frame_count,)
    _print_count_with_percentage(
        "Derived Left-hand frames",
        report.derived_left_frame_count,
        report.frame_count,)
    _print_count_with_percentage(
        "Derived Right-hand frames",
        report.derived_right_frame_count,
        report.frame_count,)
    if report.left_wrist_norm_samples > 0:
        print("Mean normalized Left wrist displacement magnitude: "
            f"{report.left_wrist_norm_sum / report.left_wrist_norm_samples:.3f}")
    if report.right_wrist_norm_samples > 0:
        print("Mean normalized Right wrist displacement magnitude: "
            f"{report.right_wrist_norm_sum / report.right_wrist_norm_samples:.3f}")
    if report.left_pinch_ratio_samples > 0:
        print("Left pinch ratio mean/min/max: "
            f"{report.left_pinch_ratio_sum / report.left_pinch_ratio_samples:.3f} / "
            f"{report.left_pinch_ratio_min:.3f} / "
            f"{report.left_pinch_ratio_max:.3f}")
    if report.right_pinch_ratio_samples > 0:
        print("Right pinch ratio mean/min/max: "
            f"{report.right_pinch_ratio_sum / report.right_pinch_ratio_samples:.3f} / "
            f"{report.right_pinch_ratio_min:.3f} / "
            f"{report.right_pinch_ratio_max:.3f}")

    _print_reason_summary("H6 body derived-geometry rejection reasons:",
        report.body_derived_reasons,
        report.frame_count,)
    _print_reason_summary("H6 Left-hand derived-geometry rejection reasons:",
        report.left_derived_reasons,
        report.frame_count,)
    _print_reason_summary("H6 Right-hand derived-geometry rejection reasons:",
        report.right_derived_reasons,
        report.frame_count,)
    print()
    print("P3 pinch-calibration summary:")
    print(f"Gesture hand: {report.gesture_hand}")
    print_sample_summary("OPEN",
        report.calibration.open_samples,
        report.calibration.open_invalid_frames,)
    print_sample_summary("CLOSED",
        report.calibration.closed_samples,
        report.calibration.closed_invalid_frames,)
    open_summary = sample_summary(report.calibration.open_samples)
    closed_summary = sample_summary(report.calibration.closed_samples)
    if open_summary is not None and closed_summary is not None:
        central_gap = open_summary["p05"] - closed_summary["p95"]
        print("P3 central separation (OPEN p05 - CLOSED p95): " f"{central_gap:.3f}")
        if central_gap > 0.0:
            print("The central labeled distributions are separated in this run.")
        else:
            print("The central labeled distributions overlap in this run.")
    print("P3 does not choose final hysteresis thresholds from this single diagnostic automatically.")
    print()
    print("H8 virtual-retargeting summary:")
    print(f"Reference captures: {report.retargeting_activation_count}")
    _print_count_with_percentage("Active retargeting frames",
        report.retargeting_active_frame_count,
        report.frame_count,)
    _print_count_with_percentage("Valid virtual target frames",
        report.retargeting_valid_target_count,
        report.frame_count,)
    print("Invalid target frames while active: "
        f"{report.retargeting_invalid_target_count}")
    print(f"Workspace-limited frames: {report.retargeting_workspace_limited_count}")
    print(f"Speed-limited frames: {report.retargeting_speed_limited_count}")
    print("Maximum normalized operator delta magnitude: "
        f"{report.max_operator_delta_norm:.3f}")
    print("Maximum virtual target displacement magnitude: "
        f"{report.max_virtual_target_norm:.3f}")
    print("H8 live coordinates are diagnostic virtual units, not physical robot coordinates.")
    print()
    print("P4E integrated pre-ROS summary:")
    if not report.p4e_enabled:
        print("P4E integrated diagnostic: DISABLED")
    else:
        print(f"P4E supervisor cycles: {report.p4e_supervisor_cycle_count}")
        print(f"P4E valid aperture frames: {report.p4e_aperture_valid_count}")
        print("P4E supervisor states: " + ", ".join(f"{state.value}={report.p4e_supervisor_state_counts[state]}" for state in SupervisorState))
        print(f"P4E arm-permitted frames: {report.p4e_motion_permitted_count}")
        print(f"P4E gripper-permitted frames: {report.p4e_gripper_permitted_count}")
        print("P4E permission invariant violations: " f"{report.p4e_permission_violation_count}")
        print("P4E robot state, transform, and controller-ready inputs were SYNTHETIC diagnostics only; no Franka state was used.")
        if report.args.p4e_auto_sequence:
            print("P4E hands-free control mode: scripted initial enable; HOLD remained fail-closed and automatic re-enable was disabled.")
            print("P4E hands-free end reason: " f"{report.p4e_auto.end_reason or 'not_completed'}")
    print()
    print(f"Average backend processing: {report.average_processing_ms:.2f} ms")
    print(f"Average complete loop rate: {report.average_loop_fps:.2f} FPS")
    print("-" * 60)

    hand_dropout_seen = (report.left_state_counts[TrackingState.DROPOUT] > 0 or report.right_state_counts[TrackingState.DROPOUT] > 0 )
    hand_lost_seen = (report.left_state_counts[TrackingState.LOST] > 0 or report.right_state_counts[TrackingState.LOST] > 0)
    print(f"Observed hand DROPOUT state: {'YES' if hand_dropout_seen else 'NO'}")
    print(f"Observed hand LOST state: {'YES' if hand_lost_seen else 'NO'}")

    if report.logger is not None:
        report.logger.write("run_summary",
            {
                "duration_s": report.duration_s,
                "frames_processed": report.frame_count,
                "raw_pose_frames": report.pose_frame_count,
                "raw_hand_frames": report.hand_frame_count,
                "two_raw_hand_frames": report.two_hand_raw_frame_count,
                "hand_landmarker_config": {
                    "min_hand_detection_confidence": report.args.hand_min_detection_confidence,
                    "min_hand_presence_confidence": report.args.hand_min_presence_confidence,
                    "min_hand_tracking_confidence": report.args.hand_min_tracking_confidence,
                },
                "hand_association": {
                    "mode": report.args.hand_association,
                    "max_distance_shoulder_widths": report.args.hand_association_max_distance_shoulder_widths,
                    "left_pose_associated_frames": report.left_pose_associated_count,
                    "right_pose_associated_frames": report.right_pose_associated_count,
                    "left_backend_label_overrides": report.left_pose_override_count,
                    "right_backend_label_overrides": report.right_pose_override_count,
                },
                "body_per_frame_valid": report.body_valid_count,
                "left_hand_per_frame_valid": report.left_hand_valid_count,
                "right_hand_per_frame_valid": report.right_hand_valid_count,
                "body_temporal_valid": report.body_temporal_valid_count,
                "left_hand_temporal_valid": report.left_temporal_valid_count,
                "right_hand_temporal_valid": report.right_temporal_valid_count,
                "smoothed_body_frames": report.smoothed_body_frame_count,
                "smoothed_left_hand_frames": report.smoothed_left_frame_count,
                "smoothed_right_hand_frames": report.smoothed_right_frame_count,
                "derived_body_frames": report.derived_body_frame_count,
                "derived_left_hand_frames": report.derived_left_frame_count,
                "derived_right_hand_frames": report.derived_right_frame_count,
                "pinch_calibration": {
                    "hand_side": report.gesture_hand,
                    "label_change_count": report.calibration.label_change_count,
                    "open_valid_samples": len(report.calibration.open_samples),
                    "open_invalid_labeled_frames": report.calibration.open_invalid_frames,
                    "open_summary": sample_summary(report.calibration.open_samples),
                    "closed_valid_samples": len(report.calibration.closed_samples),
                    "closed_invalid_labeled_frames": report.calibration.closed_invalid_frames,
                    "closed_summary": sample_summary(report.calibration.closed_samples),
                    "thresholds_selected": False,
                },
                "retargeting_activation_count": report.retargeting_activation_count,
                "retargeting_active_frames": report.retargeting_active_frame_count,
                "retargeting_valid_target_frames": report.retargeting_valid_target_count,
                "retargeting_invalid_target_frames": report.retargeting_invalid_target_count,
                "retargeting_workspace_limited_frames": report.retargeting_workspace_limited_count,
                "retargeting_speed_limited_frames": report.retargeting_speed_limited_count,
                "retargeting_max_operator_delta_norm": report.max_operator_delta_norm,
                "retargeting_max_virtual_target_norm": report.max_virtual_target_norm,
                "p4e_integrated": {
                    "enabled": report.p4e_enabled,
                    "supervisor_cycles": report.p4e_supervisor_cycle_count,
                    "aperture_valid_frames": report.p4e_aperture_valid_count,
                    "supervisor_state_counts": {
                        state.value: report.p4e_supervisor_state_counts[state]
                        for state in SupervisorState
                    },
                    "motion_permitted_frames": report.p4e_motion_permitted_count,
                    "gripper_permitted_frames": report.p4e_gripper_permitted_count,
                    "permission_invariant_violations": report.p4e_permission_violation_count,
                    "synthetic_robot_state": report.p4e_enabled,
                    "synthetic_static_transform": report.p4e_enabled,
                    "synthetic_controller_ready": report.p4e_enabled,
                    "control_mode": "hands_free_scripted" if report.args.p4e_auto_sequence else "manual_keys",
                    "auto_sequence_end_reason": report.p4e_auto.end_reason,
                },
                "average_backend_processing_ms": report.average_processing_ms,
                "average_complete_loop_fps": report.average_loop_fps,
                "body_rejection_reasons": dict(report.body_rejection_reasons),
                "left_hand_rejection_reasons": dict(report.left_hand_rejection_reasons),
                "right_hand_rejection_reasons": dict(report.right_hand_rejection_reasons),
                "body_derived_reasons": dict(report.body_derived_reasons),
                "left_derived_reasons": dict(report.left_derived_reasons),
                "right_derived_reasons": dict(report.right_derived_reasons),
            },
        )
        report.logger.close()

    if report.log_path is not None:
        print(f"H7 JSONL log saved to: {report.log_path}")

    if report.gesture_hand == "left":
        p3_selected_hand_seen = (report.left_raw_detected_count > 0 and report.left_state_counts[TrackingState.TRACKING] > 0
            and report.derived_left_frame_count > 0)
    else:
        p3_selected_hand_seen = (report.right_raw_detected_count > 0 and report.right_state_counts[TrackingState.TRACKING] > 0
            and report.derived_right_frame_count > 0)

    p3_seen = len(report.calibration.open_samples) >= 30 and len(report.calibration.closed_samples) >= 30
    if report.frame_count > 0 and p3_selected_hand_seen and p3_seen and report.log_path is not None:
        print("LIVE P3 RESULT: PASS")
        print("The live run recorded at least 30 valid labeled OPEN and CLOSED pinch samples for the selected hand.")
    else:
        print("LIVE P3 RESULT: INCOMPLETE")
        print("The pipeline ran, but P3 needs at least 30 valid labeled OPEN and CLOSED samples in the same run.")

    if report.p4e_enabled:
        p4e_live_ok = (
            report.p4e_supervisor_cycle_count > 0 and report.p4e_aperture_valid_count > 0
            and report.p4e_supervisor_state_counts[SupervisorState.ACTIVE] > 0 and report.p4e_motion_permitted_count > 0
            and report.p4e_gripper_permitted_count > 0 and report.p4e_permission_violation_count == 0
            and report.log_path is not None
        )
        if p4e_live_ok:
            print("LIVE P4E RESULT: PASS")
            print("The live pre-ROS path produced synchronized continuous aperture and"
                " Cartesian permission in ACTIVE with no command-permission invariant violation.")
        else:
            print("LIVE P4E RESULT: INCOMPLETE")
            print("P4E needs at least one valid aperture frame and at least one ACTIVE frame permitting both Cartesian and gripper "
                "outputs. Follow S -> R -> T -> wait for READY -> E.")

def main():
    args = parse_args()
    config = _build_diagnostic_configuration(args)
    _print_diagnostic_configuration(args, config)
    runtime = _open_diagnostic_runtime(args, config)

    selected_control_arm = config.selected_control_arm
    validation_config = config.validation
    temporal_config = config.temporal
    smoothing_config = config.smoothing
    derived_config = config.derived
    control_arm = config.control_arm
    gesture_hand = config.gesture_hand
    p4e_enabled = config.p4e_enabled
    gripper_aperture_config = config.aperture
    gripper_hand_side = config.gripper_hand_side
    p4e_supervisor_config = config.supervisor
    h8_config = config.h8
    h8_virtual_reference_pose = config.h8_virtual_reference_pose

    acquisition = runtime.acquisition
    backend = runtime.backend
    adapter = runtime.adapter
    tracker = runtime.tracker
    smoother = runtime.smoother
    retargeter = runtime.retargeter
    aperture_tracker = runtime.aperture_tracker
    supervisor = runtime.supervisor

    gripper_aperture_result = None
    supervisor_result = None
    supervisor_requests = _SupervisorRequestFlags()
    p4e_auto = _P4EAutoState(launch_s=time.perf_counter())

    measurement_started = False
    measurement_start_s = None

    logger = None
    log_path = None

    retargeting_result = None

    calibration = _PinchCalibrationState()

    retargeting_activation_count = 0
    retargeting_active_frame_count = 0
    retargeting_valid_target_count = 0
    retargeting_invalid_target_count = 0
    retargeting_workspace_limited_count = 0
    retargeting_speed_limited_count = 0

    max_operator_delta_norm = 0.0
    max_virtual_target_norm = 0.0

    p4e_aperture_valid_count = 0
    p4e_supervisor_cycle_count = 0
    p4e_motion_permitted_count = 0
    p4e_gripper_permitted_count = 0
    p4e_permission_violation_count = 0
    p4e_supervisor_state_counts = Counter()

    frame_count = 0

    pose_frame_count = 0
    hand_frame_count = 0

    total_hand_detections = 0
    max_hands_seen = 0
    two_hand_raw_frame_count = 0
    left_raw_detected_count = 0
    right_raw_detected_count = 0

    left_pose_associated_count = 0
    right_pose_associated_count = 0
    left_pose_override_count = 0
    right_pose_override_count = 0

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

    body_state_counts = {state: 0 for state in TrackingState}
    left_state_counts = {state: 0 for state in TrackingState}
    right_state_counts = {state: 0 for state in TrackingState}
    total_processing_s = 0.0

    try:
        _print_run_instructions(args, p4e_enabled)

        while True:
            frame = acquisition.read()
            output = backend.process(frame)
            backend_observation = adapter.convert(frame, output)
            observation = associate_hands_to_pose(backend_observation, validation_config)
            validity = evaluate_observation(observation, validation_config)
            tracking = tracker.update(observation, validity)
            smoothed = smoother.update(observation, tracking)
            derived = derive_human_kinematics(smoothed, derived_config)

            if measurement_started:
                retargeting_result = retargeter.update(derived)
            else:
                retargeting_result = None

            gripper_aperture_result = None
            supervisor_result = None
            supervisor_decision_age_s = None

            if measurement_started and p4e_enabled:
                assert aperture_tracker is not None
                assert supervisor is not None
                p4e_frame = _run_p4e_frame(aperture_tracker, supervisor, derived, tracking, retargeting_result, supervisor_requests)
                gripper_aperture_result = p4e_frame.aperture
                supervisor_result = p4e_frame.supervisor
                supervisor_decision_age_s = p4e_frame.decision_age_s

            gesture_derived_hand = selected_derived_hand(derived, gesture_hand)
            current_gesture_pinch_ratio = None if gesture_derived_hand is None else gesture_derived_hand.pinch_ratio
            pinch_calibration_frame = calibration.record_frame(gesture_hand, measurement_started, current_gesture_pinch_ratio)
            display = frame.image_bgr.copy()
            pose_landmarks_sets = output.pose_result.pose_landmarks
            hand_landmarks_sets = output.hand_result.hand_landmarks
            pose_count = len(pose_landmarks_sets)
            hand_count = len(hand_landmarks_sets)

            if measurement_started:
                frame_count += 1

                total_processing_s += output.processing_duration_s

                if not validity.body_valid_for_control:
                    body_rejection_reasons.update(validity.body_reasons)

                if pose_count > 0:
                    pose_frame_count += 1

                if hand_count > 0:
                    hand_frame_count += 1

                total_hand_detections += (hand_count)
                max_hands_seen = max(max_hands_seen, hand_count)

                if hand_count >= 2:
                    two_hand_raw_frame_count += 1

                if backend_observation.left_hands:
                    left_raw_detected_count += 1

                if backend_observation.right_hands:
                    right_raw_detected_count += 1

                if len(observation.left_hands) == 1 and observation.left_hands[0].association_source == HandAssociationSource.POSE_WRIST:
                    left_pose_associated_count += 1
                    if observation.left_hands[0].handedness.strip().lower() != "left":
                        left_pose_override_count += 1

                if len(observation.right_hands) == 1 and observation.right_hands[0].association_source == HandAssociationSource.POSE_WRIST:
                    right_pose_associated_count += 1
                    if observation.right_hands[0].handedness.strip().lower() != "right":
                        right_pose_override_count += 1

                if not validity.left_hand_valid_for_control:
                    left_hand_rejection_reasons.update(validity.left_hand_reasons)

                if not validity.right_hand_valid_for_control:
                    right_hand_rejection_reasons.update(validity.right_hand_reasons)

                if validity.available_for_visualization:
                    visualization_frame_count += 1

                if validity.body_valid_for_control:
                    body_valid_count += 1

                if validity.left_hand_valid_for_control:
                    left_hand_valid_count += 1

                if validity.right_hand_valid_for_control:
                    right_hand_valid_count += 1

                if validity.left_hand_valid_for_control and validity.right_hand_valid_for_control:
                    both_hands_valid_count += 1

                body_state_counts[tracking.body.state] += 1
                left_state_counts[tracking.left_hand.state] += 1
                right_state_counts[tracking.right_hand.state] += 1

                if tracking.body.valid_for_control:
                    body_temporal_valid_count += 1

                if tracking.left_hand.valid_for_control:
                    left_temporal_valid_count += 1

                if tracking.right_hand.valid_for_control:
                    right_temporal_valid_count += 1

                if tracking.left_hand.valid_for_control and tracking.right_hand.valid_for_control:
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

                    left_wrist = derived.body.left_wrist_displacement_normalized_body
                    right_wrist = derived.body.right_wrist_displacement_normalized_body

                    if left_wrist is not None:
                        left_wrist_norm_sum += math.sqrt(left_wrist.x * left_wrist.x + left_wrist.y * left_wrist.y + left_wrist.z * left_wrist.z)
                        left_wrist_norm_samples += 1

                    if right_wrist is not None:
                        right_wrist_norm_sum += math.sqrt(right_wrist.x * right_wrist.x + right_wrist.y * right_wrist.y + right_wrist.z * right_wrist.z)
                        right_wrist_norm_samples += 1

                else:
                    body_derived_reasons.update(derived.body_reasons)

                if derived.left_hand is not None:
                    derived_left_frame_count += 1

                    left_pinch = derived.left_hand.pinch_ratio
                    left_pinch_ratio_sum += left_pinch
                    left_pinch_ratio_samples += 1

                    left_pinch_ratio_min = (
                        left_pinch if left_pinch_ratio_min is None
                        else min(
                            left_pinch_ratio_min,
                            left_pinch,
                        )
                    )

                    left_pinch_ratio_max = (
                        left_pinch if left_pinch_ratio_max is None
                        else max(
                            left_pinch_ratio_max,
                            left_pinch,
                        )
                    )

                else:
                    left_derived_reasons.update(derived.left_hand_reasons)

                if derived.right_hand is not None:
                    derived_right_frame_count += 1

                    right_pinch = derived.right_hand.pinch_ratio
                    right_pinch_ratio_sum += right_pinch
                    right_pinch_ratio_samples += 1

                    right_pinch_ratio_min = (
                        right_pinch if right_pinch_ratio_min is None
                        else min(
                            right_pinch_ratio_min,
                            right_pinch,
                        )
                    )

                    right_pinch_ratio_max = (
                        right_pinch if right_pinch_ratio_max is None
                        else max(
                            right_pinch_ratio_max,
                            right_pinch,
                        )
                    )

                else:
                    right_derived_reasons.update(derived.right_hand_reasons)

                if retargeting_result is not None:
                    if retargeting_result.active:
                        retargeting_active_frame_count += 1

                        if retargeting_result.valid:
                            retargeting_valid_target_count += 1

                            if retargeting_result.operator_delta_normalized_body is not None:
                                delta = retargeting_result.operator_delta_normalized_body
                                delta_norm = math.sqrt(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z)
                                max_operator_delta_norm = max(max_operator_delta_norm, delta_norm)

                            if retargeting_result.target_pose is not None:
                                target = retargeting_result.target_pose.position
                                target_norm = math.sqrt(target.x * target.x + target.y * target.y + target.z * target.z)
                                max_virtual_target_norm = max(max_virtual_target_norm, target_norm)

                            if retargeting_result.workspace_limited:
                                retargeting_workspace_limited_count += 1

                            if retargeting_result.speed_limited:
                                retargeting_speed_limited_count += 1

                        else:
                            retargeting_invalid_target_count += 1

                if p4e_enabled:
                    if gripper_aperture_result is None:
                        raise RuntimeError("P4E is enabled but the aperture result is missing.")
                    if supervisor_result is None:
                        raise RuntimeError("P4E is enabled but the supervisor result is missing.")

                    p4e_supervisor_cycle_count += 1
                    p4e_supervisor_state_counts[supervisor_result.state] += 1

                    if gripper_aperture_result.measurement_valid:
                        p4e_aperture_valid_count += 1
                    if supervisor_result.motion_permitted:
                        p4e_motion_permitted_count += 1
                    if supervisor_result.gripper_command_permitted:
                        p4e_gripper_permitted_count += 1

                    if (
                        supervisor_result.state != SupervisorState.ACTIVE and (
                            supervisor_result.motion_permitted or supervisor_result.gripper_command_permitted
                            or supervisor_result.permitted_target is not None or supervisor_result.permitted_gripper_aperture is not None
                        )
                    ):
                        p4e_permission_violation_count += 1

                if logger is None:
                    raise RuntimeError("Measurement is active but H7/H8/P4E logger is not initialized.")

                logger.write("frame",
                    {
                        "measurement_frame_index": frame_count - 1,
                        "frame_sequence_id": frame.sequence_id,
                        "measurement_timestamp_s": frame.measurement_timestamp_s,
                        "image_width_px": frame.image_width_px,
                        "image_height_px": frame.image_height_px,
                        "backend_processing_duration_s": output.processing_duration_s,
                        "raw_detection_counts": {
                            "pose": pose_count,
                            "hands": hand_count,
                        },
                        "observation": observation.to_dict(),
                        "per_frame_validity": {
                            "available_for_visualization": validity.available_for_visualization,
                            "body_valid_for_control": validity.body_valid_for_control,
                            "left_hand_valid_for_control": validity.left_hand_valid_for_control,
                            "right_hand_valid_for_control": validity.right_hand_valid_for_control,
                            "body_reasons": list(validity.body_reasons),
                            "left_hand_reasons": list(validity.left_hand_reasons),
                            "right_hand_reasons": list(validity.right_hand_reasons),
                        },
                        "temporal_tracking": {
                            "body": tracking_channel_to_dict(tracking.body),
                            "left_hand": tracking_channel_to_dict(tracking.left_hand),
                            "right_hand": tracking_channel_to_dict(tracking.right_hand),
                        },
                        "smoothed_available": {
                            "body_image": smoothed.body.image_landmarks is not None,
                            "body_world": smoothed.body.world_landmarks is not None,
                            "left_hand_image": smoothed.left_hand.image_landmarks is not None,
                            "left_hand_world": smoothed.left_hand.world_landmarks is not None,
                            "right_hand_image": smoothed.right_hand.image_landmarks is not None,
                            "right_hand_world": smoothed.right_hand.world_landmarks is not None,
                        },
                        "smoothed": json_safe(smoothed),
                        "derived": derived_to_dict(derived),
                        "retargeting": retargeting_result_to_dict(retargeting_result),
                        "continuous_gripper_aperture": gripper_aperture_result_to_dict(gripper_aperture_result),
                        "supervisor": supervisor_result_to_dict(supervisor_result),
                        "synthetic_pre_ros_inputs": (
                            None if not p4e_enabled
                            else {
                                "robot_state": {
                                    "synthetic": True,
                                    "valid": True,
                                    "timestamp_s": derived.measurement_timestamp_s,
                                },
                                "transform": {
                                    "synthetic": True,
                                    "available": True,
                                    "static": True,
                                    "timestamp_s": None,
                                },
                                "controller": {
                                    "synthetic": True,
                                    "ready": True,
                                },
                            }
                        ),
                        "timing": {
                            "source_measurement_timestamp_s": frame.measurement_timestamp_s,
                            "backend_processing_duration_s": output.processing_duration_s,
                            "supervisor_decision_timestamp_s": None if supervisor_result is None else supervisor_result.decision_timestamp_s,
                            "measurement_to_supervisor_decision_s": supervisor_decision_age_s,
                        },
                        "pinch_calibration": pinch_calibration_frame,
                    },
                )

            for pose_landmarks in pose_landmarks_sets:
                draw_landmark_set(
                    image = display,
                    landmarks = pose_landmarks,
                    connections = POSE_CONNECTIONS,
                    point_color = (0, 255, 0),
                    line_color = (255, 255, 255),
                )

            for (hand_index, hand_landmarks) in enumerate(hand_landmarks_sets):
                hand_points = draw_landmark_set(
                    image=display,
                    landmarks=hand_landmarks,
                    connections=HAND_CONNECTIONS,
                    point_color=(0, 255, 255),
                    line_color=(255, 255, 255),
                )

                (handedness_name, handedness_score) = extract_handedness(output.hand_result, hand_index)
                wrist_point = (hand_points[0] if hand_points else None)

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
                    label_y = max(20, wrist_point[1] - 15)

                    draw_text(display, label, (label_x, label_y), font_scale=0.50, thickness=1)

            if smoothed.body.image_landmarks is not None:
                draw_landmark_set(
                    image = display,
                    landmarks = smoothed.body.image_landmarks.landmarks,
                    connections = POSE_CONNECTIONS,
                    point_color = SMOOTHED_COLOR,
                    line_color = SMOOTHED_COLOR,
                )

            if smoothed.left_hand.image_landmarks is not None:
                draw_landmark_set(
                    image = display,
                    landmarks = smoothed.left_hand.image_landmarks.landmarks,
                    connections=HAND_CONNECTIONS,
                    point_color=SMOOTHED_COLOR,
                    line_color=SMOOTHED_COLOR,
                )

            if smoothed.right_hand.image_landmarks is not None:
                draw_landmark_set(
                    image = display,
                    landmarks = smoothed.right_hand.image_landmarks.landmarks,
                    connections = HAND_CONNECTIONS,
                    point_color = SMOOTHED_COLOR,
                    line_color = SMOOTHED_COLOR,
                )

            processing_ms = output.processing_duration_s * 1000.0

            if measurement_started and measurement_start_s is not None:
                elapsed_s = time.perf_counter() - measurement_start_s
            else:
                elapsed_s = 0.0

            loop_fps = frame_count / elapsed_s if elapsed_s > 0.0 else 0.0
            mode_text = "MEASURING" if measurement_started else "PREVIEW - press S when ready"
            draw_text(
                display, f"Mode: {mode_text}",
                (15, 25), font_scale=0.55, thickness=2,
            )

            draw_text(display,
                (
                    f"Frame {frame.sequence_id}  "
                    f"Pose: {pose_count}  "
                    f"Hands: {hand_count}"
                ),
                (15, 50), font_scale=0.55, thickness=2,
            )

            draw_text(display,
                (
                    f"Inference: "
                    f"{processing_ms:.1f} ms  "
                    f"Loop: {loop_fps:.1f} FPS"
                ),
                (15, 75), font_scale=0.50, thickness=2,
            )

            body_quality = "OK" if validity.body_valid_for_control else "BLOCK"
            left_quality = "OK" if validity.left_hand_valid_for_control else "BLOCK"
            right_quality = "OK" if validity.right_hand_valid_for_control else "BLOCK"
            draw_text(display,
                (
                    "Frame quality - "
                    f"B:{body_quality} "
                    f"L:{left_quality} "
                    f"R:{right_quality}"
                ),
                (15, 100), font_scale=0.50, thickness=2,
            )

            draw_text(display,
                (
                    "Temporal - "
                    f"B:{tracking.body.state.value.upper()} "
                    f"L:{tracking.left_hand.state.value.upper()} "
                    f"R:{tracking.right_hand.state.value.upper()}"
                ),
                (15, 125), font_scale=0.47, thickness=2,
            )

            body_control = "GO" if tracking.body.valid_for_control else "HOLD"
            left_control = "GO" if tracking.left_hand.valid_for_control else "HOLD"
            right_control = "GO" if tracking.right_hand.valid_for_control else "HOLD"

            draw_text(display,
                (
                    "Temporal control - "
                    f"B:{body_control} "
                    f"L:{left_control} "
                    f"R:{right_control}"
                ),
                (15, 150), font_scale=0.47, thickness=2,
            )

            smoothing_status = ("Smoothing - "
                f"B:{'ON' if smoothed.body.image_landmarks is not None else 'OFF'} "
                f"L:{'ON' if smoothed.left_hand.image_landmarks is not None else 'OFF'} "
                f"R:{'ON' if smoothed.right_hand.image_landmarks is not None else 'OFF'}"
            )

            draw_text(display, smoothing_status, (15, 175), font_scale=0.47, thickness=2,)

            body_derived_status = "ON" if derived.body is not None else "OFF"

            if derived.left_hand is not None:
                left_pinch_text = f"{derived.left_hand.pinch_ratio:.2f}"
            else:
                left_pinch_text = "--"

            if derived.right_hand is not None:
                right_pinch_text = f"{derived.right_hand.pinch_ratio:.2f}"
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
                (15, 200), font_scale=0.45, thickness=2,
            )

            if derived.body is not None:
                selected_wrist = (
                    derived.body.left_wrist_displacement_normalized_body
                    if control_arm == ArmSide.LEFT else derived.body.right_wrist_displacement_normalized_body)

                if selected_wrist is not None:
                    draw_text(
                        display,
                        (
                            "H6 selected wrist/body - "
                            f"{control_arm.value.upper()}("
                            f"{selected_wrist.x:+.2f},"
                            f"{selected_wrist.y:+.2f},"
                            f"{selected_wrist.z:+.2f})"
                        ),
                        (15, 220),
                        font_scale=0.40,
                        thickness=1,
                    )

            if not measurement_started:
                h8_status = "H8 - wait for S"
            elif not retargeter.active:
                h8_status = f"H8 {control_arm.value.upper()} - INACTIVE, press R"
            elif retargeting_result is not None and retargeting_result.valid and retargeting_result.target_pose is not None:
                target = retargeting_result.target_pose.position
                h8_status = (
                    f"H8 {control_arm.value.upper()} target "
                    f"({target.x:+.2f},"
                    f"{target.y:+.2f},"
                    f"{target.z:+.2f})"
                )
            else:
                h8_status = f"H8 {control_arm.value.upper()} - NO TARGET"
            draw_text(display, h8_status, (15, 315), font_scale=0.45, thickness=2)

            if (
                retargeting_result is not None and retargeting_result.valid
                and retargeting_result.operator_delta_normalized_body is not None
            ):
                delta = retargeting_result.operator_delta_normalized_body

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

            calibration_label_text = "NONE" if calibration.label is None else calibration.label.upper()

            pinch_text = "unavailable" if current_gesture_pinch_ratio is None else f"{current_gesture_pinch_ratio:.3f}"

            draw_text(display,
                (
                    f"P3 {gesture_hand.upper()} label="
                    f"{calibration_label_text} "
                    f"pinch={pinch_text}"
                ),
                (15, 355), font_scale=0.42, thickness=2,)

            draw_text(display,
                (
                    "P3 samples "
                    f"OPEN={len(calibration.open_samples)} "
                    f"CLOSED={len(calibration.closed_samples)}"
                ),
                (15, 375), font_scale=0.40, thickness=1,)

            if p4e_enabled:
                aperture_text = "--"
                if gripper_aperture_result is not None and gripper_aperture_result.normalized_aperture is not None:
                    aperture_text = f"{gripper_aperture_result.normalized_aperture:.2f}"

                supervisor_state_text = "--" if supervisor_result is None else supervisor_result.state.value.upper()

                draw_text(display,
                    (
                        "P4E aperture="
                        f"{aperture_text} supervisor={supervisor_state_text}"
                    ),
                    (15, 395), font_scale=0.40, thickness=2,)

                if supervisor_result is not None:
                    p4e_reason = first_reason(supervisor_result.reasons)
                    draw_text(display,
                        (
                            "P4E arm="
                            f"{'PERMIT' if supervisor_result.motion_permitted else 'BLOCK'} "
                            "gripper="
                            f"{'PERMIT' if supervisor_result.gripper_command_permitted else 'BLOCK'} "
                            f"reason={p4e_reason}"
                        ),
                        (15, 415), font_scale=0.37, thickness=1,)

            if not tracking.body.valid_for_control:
                draw_text(display,
                    (
                        "B reason: "
                        f"{first_reason(tracking.body.reasons)}"
                    ),
                    (15, 245), font_scale=0.42, thickness=1,)

            if not tracking.left_hand.valid_for_control:
                draw_text(
                    display,
                    (
                        "L reason: "
                        f"{first_reason(tracking.left_hand.reasons)}"
                    ),
                    (15, 265), font_scale=0.42, thickness=1,)

            if not tracking.right_hand.valid_for_control:
                draw_text(display,
                    (
                        "R reason: "
                        f"{first_reason(tracking.right_hand.reasons)}"
                    ),
                    (15, 285), font_scale=0.42, thickness=1,)

            if p4e_enabled and args.p4e_auto_sequence:
                draw_text(display,
                    _p4e_auto_status(args, p4e_auto, measurement_started, retargeting_activation_count, supervisor_result,),
                    (15, 435), font_scale=0.34, thickness=1,)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            key = _apply_p4e_auto_key(
                args,
                p4e_auto,
                key,
                measurement_started,
                retargeting_activation_count,
                tracking,
                derived,
                gripper_aperture_result,
                retargeting_result,
                supervisor_result,
            )
            if not measurement_started and key in (ord("s"), ord("S")):
                tracker = HumanTemporalTracker(temporal_config)
                smoother = HumanLandmarkSmoother(smoothing_config)
                retargeter = RelativeRetargeter(arm_side=control_arm, config=h8_config)
                retargeting_result = None

                if aperture_tracker is not None:
                    aperture_tracker.reset()
                if supervisor is not None:
                    supervisor.reset()

                gripper_aperture_result = None
                supervisor_result = None
                supervisor_requests.clear()
                p4e_auto.reset_for_measurement()
                calibration.reset()

                logger = JsonlDiagnosticLogger(args.log_dir)
                log_path = logger.path

                measurement_started = True
                measurement_start_s = time.perf_counter()

                logger.write(
                    "run_start",
                    {
                        "created_utc": datetime.now(timezone.utc).isoformat(),
                        "camera_index": args.camera,
                        "camera_reported_properties": acquisition.get_reported_properties(),
                        "runtime": {
                            "python_version": sys.version.split()[0],
                            "platform": platform.platform(),
                            "mediapipe_version": mp.__version__,
                            "opencv_version": cv2.__version__,
                        },
                        "pose_model": file_identity(args.pose_model),
                        "hand_model": file_identity(args.hand_model),
                        "hand_landmarker_config": {
                            "min_hand_detection_confidence": args.hand_min_detection_confidence,
                            "min_hand_presence_confidence": args.hand_min_presence_confidence,
                            "min_hand_tracking_confidence": args.hand_min_tracking_confidence,
                        },
                        "validation_config": {
                            "selected_control_arm": validation_config.selected_control_arm.value,
                            "min_body_visibility": validation_config.min_body_visibility,
                            "min_body_presence": validation_config.min_body_presence,
                            "min_handedness_score": validation_config.min_handedness_score,
                            "hand_association_mode": args.hand_association,
                            "enable_pose_hand_association": validation_config.enable_pose_hand_association,
                            "max_hand_wrist_pose_distance_shoulder_widths": validation_config.max_hand_wrist_pose_distance_shoulder_widths,
                        },
                        "temporal_config": {
                            "consecutive_valid_required": temporal_config.consecutive_valid_required,
                            "dropout_timeout_s": temporal_config.dropout_timeout_s,
                        },
                        "smoothing_config": {
                            "time_constant_s": smoothing_config.time_constant_s,
                        },
                        "derived_geometry_config": {
                            "selected_control_arm": derived_config.selected_control_arm.value,
                            "min_length_model_world": derived_config.min_length_model_world,
                            "min_palm_width_to_palm_length_ratio": derived_config.min_palm_width_to_palm_length_ratio,
                            "pinch_normalization": (
                                "thumb_index_tip_distance_over_"
                                "wrist_middle_mcp_palm_length"
                            ),
                        },
                        "pinch_calibration_config": {
                            "diagnostic_only": True,
                            "hand_side": gesture_hand,
                            "open_closed_thresholds_selected": False,
                            "minimum_valid_samples_per_label_for_pass": 30,
                            "labels": ["open", "closed",],
                            "label_keys": {
                                "open": "O",
                                "closed": "C",
                                "stop_labeling": "X",
                            },
                        },
                        "retargeting_config": {
                            "diagnostic_only": True,
                            "control_arm": control_arm.value,
                            "axis_mapping": h8_config.axis_mapping.rows,
                            "scale_robot_per_normalized_body": h8_config.scale_robot_per_normalized_body,
                            "deadband_normalized_body": h8_config.deadband_normalized_body,
                            "workspace_bounds": h8_config.workspace_bounds,
                            "max_cartesian_speed_robot_per_s": h8_config.max_cartesian_speed_robot_per_s,
                            "virtual_reference_pose": h8_virtual_reference_pose,
                            "units_note": "H8 live validation uses virtual unitless target coordinates. These are not physical robot workspace units.",
                        },
                        "p4e_integrated_config": {
                            "enabled": p4e_enabled,
                            "aperture_config": (
                                None if gripper_aperture_config is None
                                else {
                                    "hand_side": gripper_hand_side.value,
                                    "closed_reference_ratio": gripper_aperture_config.closed_reference_ratio,
                                    "open_reference_ratio": gripper_aperture_config.open_reference_ratio,
                                    "reference_status": "caller_supplied_provisional_diagnostic_values",
                                }
                            ),
                            "supervisor_config": (
                                None if p4e_supervisor_config is None
                                else {
                                    **json_safe(p4e_supervisor_config),
                                    "timing_values_status": "synthetic_P1_test_values_not_Franka_limits",
                                }
                            ),
                            "synthetic_inputs": (
                                None if not p4e_enabled
                                else {
                                    "robot_state": True,
                                    "static_transform": True,
                                    "controller_ready": True,
                                    "real_Franka_state_available": False,
                                }
                            ),
                            "control_mode": (
                                "hands_free_scripted"
                                if args.p4e_auto_sequence else "manual_keys"
                            ),
                            "auto_sequence": (
                                None if not args.p4e_auto_sequence
                                else {
                                    "start_delay_s": args.p4e_auto_start_delay_s,
                                    "active_duration_s": args.p4e_auto_active_duration_s,
                                    "timing_status": "diagnostic_operator_interface_only_not_robot_limits",
                                    "auto_reenable_after_hold": False,
                                    "hold_ends_window": False,
                                    "user_q_early_stop_enabled": True,
                                    "duration_semantics": "window_after_first_ACTIVE_continues_through_HOLD",
                                }
                            ),
                            "key_controls": {
                                "capture_h8_reference": "R",
                                "request_teleoperation": "T",
                                "enable_or_reenable": "E",
                                "disable": "D",
                                "quit_early": "Q",
                            },
                        },
                        "coordinate_note": "MEDIAPIPE_WORLD values are model-world geometry, not the calibrated Part 1 workspace.",
                        "logging_note": (
                            "Each measured frame stores raw observation, validity, temporal state, full smoothed "
                            "landmarks, derived geometry, H8 retargeting state, continuous gripper aperture, P4E "
                            "supervisor state/permission/rejection reasons, P3 pinch-calibration label/sample data, and timing."
                        ),
                        "display_mirrored": False,
                        "handedness_note": (
                            "Raw MediaPipe Left/Right was physically verified to match anatomical left/right "
                            "with the current unmirrored camera pipeline."
                        ),
                    },
                )

                print("Measurement started. Only frames from this point onward will be included in the statistics.")
                print("H7 log: " f"{log_path}")

                continue

            if _handle_pinch_label_key(
                key,
                measurement_started,
                calibration,
                logger,
                derived,
                gesture_hand,
            ):
                continue
            if measurement_started and key in (ord("r"), ord("R"),):
                if derived.body is None:
                    print("H8 reference NOT captured: derived body geometry is unavailable.")
                else:
                    reference = retargeter.activate(derived, h8_virtual_reference_pose,)

                    retargeting_activation_count += 1

                    if logger is not None:
                        logger.write(
                            "retargeting_reference",
                            {
                                "activation_index": retargeting_activation_count,
                                "arm_side": reference.arm_side.value,
                                "activation_frame_sequence_id": reference.activation_frame_sequence_id,
                                "activation_timestamp_s": reference.activation_timestamp_s,
                                "human_wrist_reference_normalized_body": reference.human_wrist_reference_normalized_body,
                                "robot_pose_reference": reference.robot_pose_reference
                            },
                        )

                    print(
                        "H8 reference captured for "
                        f"{control_arm.value} arm at "
                        f"frame {reference.activation_frame_sequence_id}. "
                        "Virtual target is now relative to this pose."
                    )

                continue

            if _handle_supervisor_request_key(
                key,
                measurement_started,
                p4e_enabled,
                supervisor_requests,
                logger,
                derived,
            ):
                continue
            if key in (ord("q"), ord("Q")):
                if args.p4e_auto_sequence and p4e_auto.end_reason is None:
                    p4e_auto.end_reason = "user_q"
                    print("P4E AUTO: user requested early stop with Q. The diagnostic is ending now.")
                break

    except Exception as exc:
        print()
        print("RESULT: FAIL")
        print(f"{type(exc).__name__}: {exc}")

        if logger is not None:
            try:
                logger.write(
                    "run_error",
                    {
                        "error_type": type(exc).__name__,
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

    if measurement_started and measurement_start_s is not None:
        duration_s = time.perf_counter() - measurement_start_s
    else:
        duration_s = 0.0

    average_processing_ms = 1000.0 * total_processing_s / frame_count if frame_count > 0 else 0.0
    average_loop_fps = frame_count / duration_s if duration_s > 0.0 else 0.0

    _finalize_run(
        _RunReportData(
            args = args,
            logger = logger,
            log_path = log_path,
            gesture_hand = gesture_hand,
            calibration = calibration,
            p4e_auto = p4e_auto,
            p4e_enabled = p4e_enabled,
            measurement_started = measurement_started,
            duration_s = duration_s,
            average_processing_ms = average_processing_ms,
            average_loop_fps = average_loop_fps,
            frame_count = frame_count,
            pose_frame_count = pose_frame_count,
            hand_frame_count = hand_frame_count,
            total_hand_detections = total_hand_detections,
            max_hands_seen = max_hands_seen,
            two_hand_raw_frame_count = two_hand_raw_frame_count,
            left_raw_detected_count = left_raw_detected_count,
            right_raw_detected_count = right_raw_detected_count,
            left_pose_associated_count = left_pose_associated_count,
            right_pose_associated_count = right_pose_associated_count,
            left_pose_override_count = left_pose_override_count,
            right_pose_override_count = right_pose_override_count,
            visualization_frame_count = visualization_frame_count,
            body_valid_count = body_valid_count,
            left_hand_valid_count = left_hand_valid_count,
            right_hand_valid_count = right_hand_valid_count,
            both_hands_valid_count = both_hands_valid_count,
            body_temporal_valid_count = body_temporal_valid_count,
            left_temporal_valid_count = left_temporal_valid_count,
            right_temporal_valid_count = right_temporal_valid_count,
            both_hands_temporal_valid_count = both_hands_temporal_valid_count,
            body_state_counts = body_state_counts,
            left_state_counts = left_state_counts,
            right_state_counts = right_state_counts,
            body_rejection_reasons = body_rejection_reasons,
            left_hand_rejection_reasons = left_hand_rejection_reasons,
            right_hand_rejection_reasons = right_hand_rejection_reasons,
            smoothed_body_frame_count = smoothed_body_frame_count,
            smoothed_left_frame_count = smoothed_left_frame_count,
            smoothed_right_frame_count = smoothed_right_frame_count,
            body_separation_sum_px = body_separation_sum_px,
            body_separation_samples = body_separation_samples,
            left_separation_sum_px = left_separation_sum_px,
            left_separation_samples = left_separation_samples,
            right_separation_sum_px = right_separation_sum_px,
            right_separation_samples = right_separation_samples,
            derived_body_frame_count = derived_body_frame_count,
            derived_left_frame_count = derived_left_frame_count,
            derived_right_frame_count = derived_right_frame_count,
            left_wrist_norm_sum = left_wrist_norm_sum,
            left_wrist_norm_samples = left_wrist_norm_samples,
            right_wrist_norm_sum = right_wrist_norm_sum,
            right_wrist_norm_samples = right_wrist_norm_samples,
            left_pinch_ratio_sum = left_pinch_ratio_sum,
            left_pinch_ratio_samples = left_pinch_ratio_samples,
            left_pinch_ratio_min = left_pinch_ratio_min,
            left_pinch_ratio_max = left_pinch_ratio_max,
            right_pinch_ratio_sum = right_pinch_ratio_sum,
            right_pinch_ratio_samples = right_pinch_ratio_samples,
            right_pinch_ratio_min = right_pinch_ratio_min,
            right_pinch_ratio_max = right_pinch_ratio_max,
            body_derived_reasons = body_derived_reasons,
            left_derived_reasons = left_derived_reasons,
            right_derived_reasons = right_derived_reasons,
            retargeting_activation_count = retargeting_activation_count,
            retargeting_active_frame_count = retargeting_active_frame_count,
            retargeting_valid_target_count = retargeting_valid_target_count,
            retargeting_invalid_target_count = retargeting_invalid_target_count,
            retargeting_workspace_limited_count = retargeting_workspace_limited_count,
            retargeting_speed_limited_count = retargeting_speed_limited_count,
            max_operator_delta_norm = max_operator_delta_norm,
            max_virtual_target_norm = max_virtual_target_norm,
            p4e_supervisor_cycle_count = p4e_supervisor_cycle_count,
            p4e_aperture_valid_count = p4e_aperture_valid_count,
            p4e_supervisor_state_counts = p4e_supervisor_state_counts,
            p4e_motion_permitted_count = p4e_motion_permitted_count,
            p4e_gripper_permitted_count = p4e_gripper_permitted_count,
            p4e_permission_violation_count = p4e_permission_violation_count,
        )
    )

if __name__ == "__main__":
    main()