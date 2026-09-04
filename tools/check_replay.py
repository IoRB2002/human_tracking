from __future__ import annotations
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import tempfile

from human_tracking.gripper_intent import (
    GripperApertureConfig,
    GripperApertureTracker,
    HandSide,
)
from human_tracking.observation import (
    ControlArm,
    CoordinateSpace,
    HandObservation,
    HumanObservation,
    Landmark,
    LandmarkSet,
    ObservationValidationConfig,
    evaluate_observation,
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
from human_tracking.supervisor import (
    CommandSupervisor,
    CommandSupervisorConfig,
    GripperLossPolicy,
    SupervisorCycleInput,
    SupervisorState,
)
from human_tracking.tracking import (
    DerivedGeometryConfig,
    HumanLandmarkSmoother,
    HumanTemporalTracker,
    LandmarkSmoothingConfig,
    TemporalTrackingConfig,
    Vector3,
    derive_human_kinematics,
)

@dataclass(frozen=True)
class ReplayStep:
    label: str
    sequence_id: int
    measurement_timestamp_s: float
    wrist_offset_x: float = 0.0
    pinch_ratio: float = 0.80
    body_present: bool = True
    left_hand_present: bool = True
    capture_reference: bool = False
    deactivate_retargeter: bool = False
    activation_requested: bool = False
    enable_requested: bool = False
    disable_requested: bool = False
    fault_reset_requested: bool = False
    controller_fault: bool = False
    decision_delay_s: float = 0.01

@dataclass(frozen=True)
class ReplayFrameResult:
    label: str
    sequence_id: int
    measurement_timestamp_s: float
    body_tracking_state: str
    left_hand_tracking_state: str
    retargeting_active: bool
    retargeting_valid: bool
    aperture_valid: bool
    supervisor_state: str
    motion_permitted: bool
    gripper_command_permitted: bool
    supervisor_reasons: tuple[str, ...]
    gripper_reasons: tuple[str, ...]

_BODY_KEY_POSITIONS = {
    11: (-0.20, 1.00, 0.00),
    12: (0.20, 1.00, 0.00),
    13: (-0.40, 0.80, 0.00),
    15: (-0.60, 0.60, 0.00),
    23: (-0.15, 0.00, 0.00),
    24: (0.15, 0.00, 0.00),
}

# Check if the following values are Franka aproprate
_OBSERVATION_CONFIG = ObservationValidationConfig(
    selected_control_arm = ControlArm.LEFT,
    min_body_visibility = 0.50,
    min_body_presence = 0.50,
    min_handedness_score = 0.50,
)
_TEMPORAL_CONFIG = TemporalTrackingConfig(
    consecutive_valid_required = 3,
    dropout_timeout_s = 0.25,
)
_SMOOTHING_CONFIG = LandmarkSmoothingConfig(
    time_constant_s = 0.10,
)
_DERIVED_CONFIG = DerivedGeometryConfig(
    selected_control_arm = ControlArm.LEFT,
    min_length_model_world = 1e-5,
)
_APERTURE_CONFIG = GripperApertureConfig(
    closed_reference_ratio = 0.40,
    open_reference_ratio = 1.20,
)
_RETARGETING_CONFIG = RetargetingConfig(
    axis_mapping = AxisMapping.identity(),
    scale_robot_per_normalized_body = Vector3(1.0, 1.0, 1.0),
    deadband_normalized_body = Vector3(0.0, 0.0, 0.0),
    workspace_bounds = CartesianBounds(
        minimum=Vector3(-10.0, -10.0, -10.0),
        maximum=Vector3(10.0, 10.0, 10.0),
    ),
    max_cartesian_speed_robot_per_s=None,
)
_SUPERVISOR_CONFIG = CommandSupervisorConfig(
    consecutive_valid_required = 3,
    max_human_age_s = 0.20,
    max_robot_state_age_s = 0.20,
    max_dynamic_transform_age_s = 0.20,
    max_target_age_s = 0.20,
    max_human_robot_skew_s = 0.05,
    max_human_transform_skew_s = 0.05,
    max_human_target_skew_s = 0.01,
    gripper_loss_policy = GripperLossPolicy.HOLD_TELEOP,
)
_SYNTHETIC_ROBOT_REFERENCE = RobotAgnosticPose(
    position = Vector3(0.0, 0.0, 0.0),
    orientation_xyzw = Quaternion(0.0, 0.0, 0.0, 1.0),
)

def _landmark_set(
    count: int,
    coordinate_space: CoordinateSpace,
    positions: dict[int, tuple[float, float, float]],
    *,
    with_confidence: bool,
) -> LandmarkSet:
    landmarks: list[Landmark] = []

    for index in range(count):
        x, y, z = positions.get(index, (0.001 * index, 0.002 * index, 0.0))
        landmarks.append(
            Landmark(
                index = index,
                name = f"landmark_{index}",
                x = x,
                y = y,
                z = z,
                visibility = 0.95 if with_confidence else None,
                presence = 0.95 if with_confidence else None,
            )
        )

    return LandmarkSet(
        coordinate_space = coordinate_space,
        landmarks=tuple(landmarks),
    )


def _body_sets(wrist_offset_x: float) -> tuple[LandmarkSet, LandmarkSet]:
    positions = dict(_BODY_KEY_POSITIONS)
    wrist = positions[15]
    positions[15] = (
        wrist[0] + wrist_offset_x,
        wrist[1],
        wrist[2],
    )

    return (
        _landmark_set(
            33,
            CoordinateSpace.NORMALIZED_IMAGE,
            positions,
            with_confidence=True,
        ),
        _landmark_set(
            33,
            CoordinateSpace.MEDIAPIPE_WORLD,
            positions,
            with_confidence=True,
        ),
    )

def _left_hand(pinch_ratio: float) -> HandObservation:
    positions = {
        0: (0.0, 0.0, 0.0),
        4: (0.40 + pinch_ratio, 1.50, 0.0),
        5: (0.50, 1.00, 0.0),
        8: (0.40, 1.50, 0.0),
        17: (-0.50, 1.00, 0.0),
    }

    return HandObservation(
        handedness = "Left",
        handedness_score = 0.99,
        image_landmarks = _landmark_set(
            21,
            CoordinateSpace.NORMALIZED_IMAGE,
            positions,
            with_confidence = False,
        ),
        world_landmarks = _landmark_set(
            21,
            CoordinateSpace.MEDIAPIPE_WORLD,
            positions,
            with_confidence = False,
        ),
    )

def _observation(step: ReplayStep) -> HumanObservation:
    if step.body_present:
        body_image, body_world = _body_sets(step.wrist_offset_x)
    else:
        body_image = None
        body_world = None

    left_hands = (
        (_left_hand(step.pinch_ratio),)
        if step.left_hand_present
        else ()
    )

    return HumanObservation(
        frame_sequence_id = step.sequence_id,
        measurement_timestamp_s = step.measurement_timestamp_s,
        image_width_px = 640,
        image_height_px = 480,
        body_image_landmarks = body_image,
        body_world_landmarks = body_world,
        left_hands = left_hands,
        right_hands = (),
        unknown_hands = (),
        backend_name = "synthetic_replay",
        backend_version = "p4f",
    )

def _scenario() -> list[ReplayStep]:
    steps: list[ReplayStep] = []
    sequence_id = 0
    timestamp_s = 10.0

    def add(label: str, *, dt: float = 0.10, **kwargs) -> None:
        nonlocal sequence_id, timestamp_s
        sequence_id += 1
        timestamp_s += dt
        steps.append(
            ReplayStep(
                label = label,
                sequence_id = sequence_id,
                measurement_timestamp_s = round(timestamp_s, 6),
                **kwargs,
            )
        )

    # Initial temporal acquisition and H8 reference capture.
    add("initial_acquiring_1")
    add("initial_acquiring_2")
    add("reference_capture", capture_reference=True)

    # Supervisor activation gate: ACQUIRING -> READY -> explicit enable.
    add("activation_requested", activation_requested=True)
    add("supervisor_acquiring_1", wrist_offset_x = 0.01)
    add("supervisor_acquiring_2", wrist_offset_x = 0.02)
    add("ready", wrist_offset_x = 0.03)
    add("active_initial", wrist_offset_x = 0.04, enable_requested = True)
    add("active_motion", wrist_offset_x = 0.06, pinch_ratio = 0.95)

    # Body tracking dropout must invalidate H8 and send ACTIVE -> HOLD.
    add("body_dropout_hold", body_present=False, pinch_ratio = 0.90)
    add("body_reacquire_1", wrist_offset_x = 0.06)
    add("body_reacquire_2", wrist_offset_x = 0.06)
    add("body_reacquire_3", wrist_offset_x = 0.06)
    add("body_hold_stable_2", wrist_offset_x = 0.06)
    add("body_recovered_wait_enable", wrist_offset_x = 0.06)
    add("body_reenabled_active", wrist_offset_x = 0.06, enable_requested = True)

    # Selected-hand loss under HOLD_TELEOP suppresses both command channels.
    add("hand_loss_hold", wrist_offset_x = 0.07, left_hand_present = False)
    add("hand_reacquire_1", wrist_offset_x = 0.07)
    add("hand_reacquire_2", wrist_offset_x = 0.07)
    add("hand_reacquire_3", wrist_offset_x = 0.07)
    add("hand_hold_stable_2", wrist_offset_x = 0.07)
    add("hand_recovered_wait_enable", wrist_offset_x = 0.07)
    add("hand_reenabled_active", wrist_offset_x = 0.07, enable_requested = True)

    # A late supervisory decision makes otherwise valid data stale.
    add("stale_data_hold", wrist_offset_x = 0.08, decision_delay_s = 0.25)
    # Jump forward enough that the next supervisor decision timestamp remains
    # strictly increasing after the deliberately late decision above.
    add("stale_recovery_1", dt = 0.30, wrist_offset_x = 0.08)
    add("stale_recovery_2", wrist_offset_x = 0.08)
    add("stale_recovery_3", wrist_offset_x = 0.08)
    add("stale_recovered_wait_enable", wrist_offset_x = 0.08)
    add("stale_reenabled_active", wrist_offset_x = 0.08, enable_requested = True)

    # Explicit H8 deactivation is a separate retargeting invalidation event.
    add("retargeting_invalid_hold", wrist_offset_x = 0.09, deactivate_retargeter = True)
    add("retarget_reference_recapture", wrist_offset_x = 0.09, capture_reference = True)
    add("retarget_recovery_1", wrist_offset_x = 0.10)
    add("retarget_recovery_2", wrist_offset_x = 0.11)
    add("retarget_recovery_3", wrist_offset_x = 0.12)
    add("retarget_recovered_wait_enable", wrist_offset_x = 0.12)
    add("retarget_reenabled_active", wrist_offset_x = 0.12, enable_requested=True)

    # Controller fault latches FAULT until an explicit reset request.
    add("fault_latched", wrist_offset_x = 0.12, controller_fault = True)
    add("fault_requires_reset", wrist_offset_x = 0.12)
    add("fault_reset_disabled", wrist_offset_x = 0.12, fault_reset_requested = True)

    return steps

def _write_and_readback(steps: list[ReplayStep]) -> list[ReplayStep]:
    with tempfile.TemporaryDirectory(prefix = "human_tracking_p4f_") as temp_dir:
        replay_path = Path(temp_dir) / "synthetic_replay.jsonl"

        with replay_path.open("w", encoding="utf-8") as stream:
            for step in steps:
                stream.write(json.dumps(asdict(step), sort_keys=True))
                stream.write("\n")

        loaded: list[ReplayStep] = []
        with replay_path.open("r", encoding="utf-8") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    loaded.append(ReplayStep(**payload))
                except (json.JSONDecodeError, TypeError) as exc:
                    raise ValueError(f"Invalid replay JSONL record at line {line_number}.") from exc

    if loaded != steps:
        raise AssertionError("Replay JSONL readback changed the synthetic tape.")

    return loaded

def _validate_replay_order(steps: list[ReplayStep]) -> None:
    previous_sequence_id: int | None = None
    previous_timestamp_s: float | None = None

    for step in steps:
        if step.sequence_id < 0:
            raise ValueError("Replay sequence IDs must be non-negative.")
        if step.measurement_timestamp_s < 0.0:
            raise ValueError("Replay timestamps must be non-negative.")
        if previous_sequence_id is not None and step.sequence_id <= previous_sequence_id:
            raise ValueError("Replay sequence IDs must be strictly increasing.")
        if previous_timestamp_s is not None and step.measurement_timestamp_s <= previous_timestamp_s:
            raise ValueError("Replay timestamps must be strictly increasing.")

        previous_sequence_id = step.sequence_id
        previous_timestamp_s = step.measurement_timestamp_s


def _run_replay(steps: list[ReplayStep]) -> list[ReplayFrameResult]:
    _validate_replay_order(steps)

    temporal_tracker = HumanTemporalTracker(_TEMPORAL_CONFIG)
    smoother = HumanLandmarkSmoother(_SMOOTHING_CONFIG)
    aperture_tracker = GripperApertureTracker(
        hand_side = HandSide.LEFT,
        config = _APERTURE_CONFIG,
    )
    retargeter = RelativeRetargeter(
        arm_side = ArmSide.LEFT,
        config = _RETARGETING_CONFIG,
    )
    supervisor = CommandSupervisor(_SUPERVISOR_CONFIG)

    outputs: list[ReplayFrameResult] = []

    for step in steps:
        observation = _observation(step)
        validity = evaluate_observation(observation, _OBSERVATION_CONFIG)
        tracking = temporal_tracker.update(observation, validity)
        smoothed = smoother.update(observation, tracking)
        derived = derive_human_kinematics(smoothed, _DERIVED_CONFIG)
        aperture = aperture_tracker.update(derived)

        if step.deactivate_retargeter:
            retargeter.deactivate()

        if step.capture_reference:
            if derived.body is None:
                raise AssertionError( f"Replay step {step.label!r} could not capture H8 reference.")
            retargeter.activate(derived, _SYNTHETIC_ROBOT_REFERENCE)
            retargeting_result = None
        else:
            retargeting_result = retargeter.update(derived)

        decision_timestamp_s = step.measurement_timestamp_s + step.decision_delay_s

        supervisor_result = supervisor.update(
            SupervisorCycleInput(
                cycle_sequence_id = step.sequence_id,
                decision_timestamp_s = decision_timestamp_s,
                human_valid = tracking.body.valid_for_control and derived.body is not None,
                human_frame_sequence_id = step.sequence_id,
                human_measurement_timestamp_s = step.measurement_timestamp_s,
                robot_state_valid = True,
                robot_state_timestamp_s = step.measurement_timestamp_s,
                transform_available = True,
                transform_timestamp_s = None,
                retargeting_result = retargeting_result,
                controller_ready = True,
                gripper_aperture_result = aperture,
                activation_requested = step.activation_requested,
                enable_requested = step.enable_requested,
                disable_requested = step.disable_requested,
                fault_reset_requested = step.fault_reset_requested,
                controller_fault = step.controller_fault,
            )
        )

        if supervisor_result.state != SupervisorState.ACTIVE:
            if supervisor_result.motion_permitted:
                raise AssertionError(f"Motion was permitted outside ACTIVE at {step.label}.")
            if supervisor_result.permitted_target is not None:
                raise AssertionError(f"A Cartesian target escaped outside ACTIVE at {step.label}.")
            if supervisor_result.gripper_command_permitted:
                raise AssertionError(f"Gripper motion was permitted outside ACTIVE at {step.label}.")
            if supervisor_result.permitted_gripper_aperture is not None:
                raise AssertionError(f"A gripper aperture escaped outside ACTIVE at {step.label}.")

        outputs.append(
            ReplayFrameResult(
                label = step.label,
                sequence_id = step.sequence_id,
                measurement_timestamp_s = step.measurement_timestamp_s,
                body_tracking_state = tracking.body.state.value,
                left_hand_tracking_state = tracking.left_hand.state.value,
                retargeting_active = False if retargeting_result is None else retargeting_result.active,
                retargeting_valid = False if retargeting_result is None else retargeting_result.valid,
                aperture_valid = aperture.measurement_valid,
                supervisor_state = supervisor_result.state.value,
                motion_permitted = supervisor_result.motion_permitted,
                gripper_command_permitted = supervisor_result.gripper_command_permitted,
                supervisor_reasons = supervisor_result.reasons,
                gripper_reasons = supervisor_result.gripper_reasons,
            )
        )

    return outputs


def _by_label(outputs: list[ReplayFrameResult]) -> dict[str, ReplayFrameResult]:
    return {output.label: output for output in outputs}

def _assert_state(
    record: ReplayFrameResult,
    expected_state: SupervisorState,
    *,
    permitted: bool,
) -> None:
    if record.supervisor_state != expected_state.value:
        raise AssertionError(
            f"{record.label}: expected {expected_state.value}, "
            f"received {record.supervisor_state}."
        )

    if record.motion_permitted != permitted:
        raise AssertionError(f"{record.label}: unexpected Cartesian permission state.")

    if record.gripper_command_permitted != permitted:
        raise AssertionError(f"{record.label}: unexpected gripper permission state.")

def main() -> None:
    print("=" * 60)
    print("P4F - Deterministic Non-ROS Replay/Readback Check")
    print("=" * 60)
    print("All replay workspace/timing/aperture values are SYNTHETIC regression parameters, not Franka limits or calibration values.")

    scenario = _scenario()
    replay = _write_and_readback(scenario)
    print("P4F synthetic JSONL write/readback: PASS")

    _validate_replay_order(replay)
    print("P4F replay monotonic sequence/timestamps: PASS")

    first_outputs = _run_replay(replay)
    second_outputs = _run_replay(replay)
    if first_outputs != second_outputs:
        raise AssertionError("Identical P4F replay inputs produced different outputs.")
    print("P4F deterministic repeatability: PASS")

    records = _by_label(first_outputs)

    _assert_state(records["ready"], SupervisorState.READY, permitted = False)
    _assert_state(records["active_initial"], SupervisorState.ACTIVE, permitted = True)

    body_dropout = records["body_dropout_hold"]
    _assert_state(body_dropout, SupervisorState.HOLD, permitted = False)
    if body_dropout.body_tracking_state != "dropout":
        raise AssertionError("P4F did not exercise the temporal body DROPOUT state.")
    if body_dropout.retargeting_valid:
        raise AssertionError("P4F body dropout did not invalidate H8 retargeting.")
    print("P4F tracking dropout + retargeting invalidation -> HOLD: PASS")

    _assert_state(
        records["body_recovered_wait_enable"],
        SupervisorState.HOLD,
        permitted = False,
    )
    if "explicit_reenable_required" not in records["body_recovered_wait_enable"].supervisor_reasons:
        raise AssertionError("P4F body recovery bypassed explicit re-enable.")
    _assert_state(
        records["body_reenabled_active"],
        SupervisorState.ACTIVE,
        permitted = True,
    )
    print("P4F body recovery explicit re-enable: PASS")

    hand_loss = records["hand_loss_hold"]
    _assert_state(hand_loss, SupervisorState.HOLD, permitted = False)
    if hand_loss.aperture_valid:
        raise AssertionError("P4F hand-loss frame unexpectedly retained aperture validity.")
    if "gripper_aperture_invalid" not in hand_loss.supervisor_reasons:
        raise AssertionError("P4F HOLD_TELEOP did not reject invalid gripper data.")
    print("P4F gripper-hand loss -> HOLD: PASS")

    _assert_state(
        records["hand_recovered_wait_enable"],
        SupervisorState.HOLD,
        permitted = False,
    )
    if "explicit_reenable_required" not in records["hand_recovered_wait_enable"].supervisor_reasons:
        raise AssertionError("P4F hand recovery bypassed explicit re-enable.")
    _assert_state(
        records["hand_reenabled_active"],
        SupervisorState.ACTIVE,
        permitted = True,
    )
    print("P4F hand recovery explicit re-enable: PASS")

    stale = records["stale_data_hold"]
    _assert_state(stale, SupervisorState.HOLD, permitted = False)
    if "human_stale" not in stale.supervisor_reasons:
        raise AssertionError("P4F stale-data frame did not report human_stale.")
    print("P4F stale-data fail-closed handling: PASS")

    _assert_state(
        records["stale_recovered_wait_enable"],
        SupervisorState.HOLD,
        permitted = False,
    )
    _assert_state(
        records["stale_reenabled_active"],
        SupervisorState.ACTIVE,
        permitted = True,
    )
    print("P4F stale-data recovery explicit re-enable: PASS")

    retarget_invalid = records["retargeting_invalid_hold"]
    _assert_state(retarget_invalid, SupervisorState.HOLD, permitted = False)
    if retarget_invalid.retargeting_valid:
        raise AssertionError("P4F explicit H8 deactivation remained retargeting-valid.")
    if "retargeting_inactive" not in retarget_invalid.supervisor_reasons:
        raise AssertionError("P4F explicit H8 invalidation was not rejected.")
    print("P4F explicit retargeting invalidation -> HOLD: PASS")

    _assert_state(
        records["retarget_recovered_wait_enable"],
        SupervisorState.HOLD,
        permitted = False,
    )
    _assert_state(
        records["retarget_reenabled_active"],
        SupervisorState.ACTIVE,
        permitted = True,
    )
    print("P4F retargeting recovery explicit re-enable: PASS")

    _assert_state(records["fault_latched"], SupervisorState.FAULT, permitted = False)
    _assert_state(
        records["fault_requires_reset"],
        SupervisorState.FAULT,
        permitted = False,
    )
    if "fault_reset_required" not in records["fault_requires_reset"].supervisor_reasons:
        raise AssertionError("P4F FAULT did not remain latched without manual reset.")
    _assert_state(
        records["fault_reset_disabled"],
        SupervisorState.DISABLED,
        permitted = False,
    )
    print("P4F FAULT latch/manual reset: PASS")

    for output in first_outputs:
        if output.supervisor_state != SupervisorState.ACTIVE.value:
            if output.motion_permitted or output.gripper_command_permitted:
                raise AssertionError(f"P4F command escaped outside ACTIVE at {output.label}.")
    print("P4F no command outside ACTIVE: PASS")

    duplicate_sequence = list(replay)
    duplicate_sequence[1] = ReplayStep(
        **{
            **asdict(duplicate_sequence[1]),
            "sequence_id": duplicate_sequence[0].sequence_id,
        }
    )
    try:
        _validate_replay_order(duplicate_sequence)
    except ValueError:
        pass
    else:
        raise AssertionError("P4F accepted a duplicate replay sequence ID.")

    duplicate_timestamp = list(replay)
    duplicate_timestamp[1] = ReplayStep(
        **{
            **asdict(duplicate_timestamp[1]),
            "measurement_timestamp_s": duplicate_timestamp[0].measurement_timestamp_s,
        }
    )
    try:
        _validate_replay_order(duplicate_timestamp)
    except ValueError:
        pass
    else:
        raise AssertionError("P4F accepted a duplicate replay timestamp.")
    print("P4F invalid replay ordering rejection: PASS")

    print("-" * 60)
    print("P4F REPLAY RESULT: PASS")
    print(
        "The transport-independent observation -> tracking -> smoothing -> H6 -> "
        "H8/aperture -> supervisor path replayed deterministically and failed "
        "closed on dropout, stale data, hand loss, retargeting invalidation, and fault."
    )

if __name__ == "__main__":
    main()