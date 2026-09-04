from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import math
from human_tracking.gripper_intent import GripperApertureResult
from human_tracking.retargeting import RetargetingResult, RobotAgnosticPose

class SupervisorState(str, Enum):
    DISABLED = "disabled"
    ACQUIRING = "acquiring"
    READY = "ready"
    ACTIVE = "active"
    HOLD = "hold"
    FAULT = "fault"

class GripperLossPolicy(str, Enum):
    HOLD_TELEOP = "hold_teleop"
    ALLOW_ARM_ONLY = "allow_arm_only"


@dataclass(frozen=True)
class CommandSupervisorConfig:
    consecutive_valid_required: int
    max_human_age_s: float
    max_robot_state_age_s: float
    max_dynamic_transform_age_s: float
    max_target_age_s: float
    max_human_robot_skew_s: float
    max_human_transform_skew_s: float
    max_human_target_skew_s: float
    gripper_loss_policy: GripperLossPolicy | None = None

    def __post_init__(self) -> None:
        if self.consecutive_valid_required < 1:
            raise ValueError("consecutive_valid_required must be at least 1.")

        numeric_limits = (
            ("max_human_age_s", self.max_human_age_s),
            ("max_robot_state_age_s", self.max_robot_state_age_s),
            ("max_dynamic_transform_age_s", self.max_dynamic_transform_age_s),
            ("max_target_age_s", self.max_target_age_s),
            ("max_human_robot_skew_s", self.max_human_robot_skew_s),
            ("max_human_transform_skew_s", self.max_human_transform_skew_s),
            ("max_human_target_skew_s", self.max_human_target_skew_s),
        )

        for name, value in numeric_limits:
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative.")

        if self.gripper_loss_policy is not None and not isinstance(self.gripper_loss_policy, GripperLossPolicy):
            raise ValueError("gripper_loss_policy must be a GripperLossPolicy or None.")

@dataclass(frozen=True)
class SupervisorCycleInput:
    cycle_sequence_id: int
    decision_timestamp_s: float
    human_valid: bool
    human_frame_sequence_id: int | None
    human_measurement_timestamp_s: float | None
    robot_state_valid: bool
    robot_state_timestamp_s: float | None
    transform_available: bool
    transform_timestamp_s: float | None
    retargeting_result: RetargetingResult | None
    controller_ready: bool
    gripper_aperture_result: GripperApertureResult | None = None
    activation_requested: bool = False
    enable_requested: bool = False
    disable_requested: bool = False
    fault_reset_requested: bool = False
    controller_fault: bool = False
    persistent_inconsistency: bool = False
    external_stop: bool = False

@dataclass(frozen=True)
class CommandSupervisorResult:
    cycle_sequence_id: int
    decision_timestamp_s: float
    previous_state: SupervisorState
    state: SupervisorState
    state_changed: bool
    motion_permitted: bool
    permitted_target: RobotAgnosticPose | None
    gripper_command_permitted: bool
    permitted_gripper_aperture: float | None
    consecutive_valid_cycles: int
    reasons: tuple[str, ...]
    gripper_reasons: tuple[str, ...]

class CommandSupervisor:
    def __init__(self, config: CommandSupervisorConfig):
        self.config = config
        self._state = SupervisorState.DISABLED
        self._consecutive_valid_cycles = 0
        self._last_cycle_sequence_id: int | None = None
        self._last_decision_timestamp_s: float | None = None

    @property
    def state(self) -> SupervisorState:
        return self._state

    def reset(self) -> None:
        self._state = SupervisorState.DISABLED
        self._consecutive_valid_cycles = 0
        self._last_cycle_sequence_id = None
        self._last_decision_timestamp_s = None

    def update(self, cycle: SupervisorCycleInput) -> CommandSupervisorResult:
        self._validate_cycle_order(cycle)
        previous_state = self._state
        self._last_cycle_sequence_id = cycle.cycle_sequence_id
        self._last_decision_timestamp_s = cycle.decision_timestamp_s
        fault_reasons = self._fault_reasons(cycle)

        if fault_reasons:
            self._state = SupervisorState.FAULT
            self._consecutive_valid_cycles = 0
            return self._result(
                cycle = cycle,
                previous_state = previous_state,
                motion_permitted = False,
                permitted_target = None,
                reasons = fault_reasons
            )

        if self._state == SupervisorState.FAULT:
            if cycle.fault_reset_requested:
                self._state = SupervisorState.DISABLED
                self._consecutive_valid_cycles = 0
                return self._result(
                    cycle = cycle,
                    previous_state = previous_state,
                    motion_permitted = False,
                    permitted_target = None,
                    reasons = ("teleoperation_disabled",)
                )

            return self._result(
                cycle = cycle,
                previous_state = previous_state,
                motion_permitted = False,
                permitted_target = None,
                reasons = ("fault_reset_required",)
            )

        if cycle.disable_requested:
            self._state = SupervisorState.DISABLED
            self._consecutive_valid_cycles = 0
            return self._result(
                cycle = cycle,
                previous_state = previous_state,
                motion_permitted = False,
                permitted_target = None,
                reasons = ("teleoperation_disabled",)
            )

        required_reasons = self._required_validity_reasons(cycle)

        if self._state == SupervisorState.DISABLED:
            self._consecutive_valid_cycles = 0

            if cycle.activation_requested:
                self._state = SupervisorState.ACQUIRING
                return self._result(
                    cycle = cycle,
                    previous_state = previous_state,
                    motion_permitted = False,
                    permitted_target = None,
                    reasons = ("stability_acquiring",)
                )

            return self._result(
                cycle = cycle,
                previous_state = previous_state,
                motion_permitted = False,
                permitted_target = None,
                reasons = ("teleoperation_disabled",)
            )

        if self._state == SupervisorState.ACQUIRING:
            if required_reasons:
                self._consecutive_valid_cycles = 0
                return self._result(
                    cycle = cycle,
                    previous_state = previous_state,
                    motion_permitted = False,
                    permitted_target = None,
                    reasons = required_reasons
                )

            self._consecutive_valid_cycles += 1

            if self._consecutive_valid_cycles >= self.config.consecutive_valid_required:
                self._state = SupervisorState.READY
                return self._result(
                    cycle = cycle,
                    previous_state = previous_state,
                    motion_permitted = False,
                    permitted_target = None,
                    reasons = ("awaiting_enable",)
                )

            return self._result(
                cycle = cycle,
                previous_state = previous_state,
                motion_permitted = False,
                permitted_target = None,
                reasons = ("stability_acquiring",)
            )

        if self._state == SupervisorState.READY:
            if required_reasons:
                self._state = SupervisorState.ACQUIRING
                self._consecutive_valid_cycles = 0
                return self._result(
                    cycle = cycle,
                    previous_state = previous_state,
                    motion_permitted = False,
                    permitted_target = None,
                    reasons = required_reasons
                )

            if cycle.enable_requested:
                target_reasons = self._target_validity_reasons(cycle)
                if target_reasons:
                    return self._result(
                        cycle = cycle,
                        previous_state = previous_state,
                        motion_permitted = False,
                        permitted_target = None,
                        reasons = target_reasons
                    )

                assert cycle.retargeting_result is not None
                assert cycle.retargeting_result.target_pose is not None

                self._state = SupervisorState.ACTIVE
                return self._result(
                    cycle = cycle,
                    previous_state = previous_state,
                    motion_permitted = True,
                    permitted_target = cycle.retargeting_result.target_pose,
                    reasons = ()
                )

            return self._result(
                cycle = cycle,
                previous_state = previous_state,
                motion_permitted = False,
                permitted_target = None,
                reasons = ("awaiting_enable",)
            )

        if self._state == SupervisorState.ACTIVE:
            target_reasons = self._target_validity_reasons(cycle)
            active_reasons = _merge_reasons(required_reasons, target_reasons)

            if active_reasons:
                self._state = SupervisorState.HOLD
                self._consecutive_valid_cycles = 0
                return self._result(
                    cycle = cycle,
                    previous_state = previous_state,
                    motion_permitted = False,
                    permitted_target = None,
                    reasons = active_reasons
                )

            assert cycle.retargeting_result is not None
            assert cycle.retargeting_result.target_pose is not None

            return self._result(
                cycle = cycle,
                previous_state = previous_state,
                motion_permitted = True,
                permitted_target = cycle.retargeting_result.target_pose,
                reasons = ()
            )

        if self._state == SupervisorState.HOLD:
            target_reasons = self._target_validity_reasons(cycle)
            recovery_reasons = _merge_reasons(required_reasons, target_reasons)

            if recovery_reasons:
                self._consecutive_valid_cycles = 0
                return self._result(
                    cycle = cycle,
                    previous_state = previous_state,
                    motion_permitted = False,
                    permitted_target = None,
                    reasons = recovery_reasons
                )

            self._consecutive_valid_cycles += 1
            stable = self._consecutive_valid_cycles >= self.config.consecutive_valid_required

            if stable and cycle.enable_requested:
                assert cycle.retargeting_result is not None
                assert cycle.retargeting_result.target_pose is not None

                self._state = SupervisorState.ACTIVE
                return self._result(
                    cycle = cycle,
                    previous_state = previous_state,
                    motion_permitted = True,
                    permitted_target = cycle.retargeting_result.target_pose,
                    reasons = ()
                )

            if stable:
                return self._result(
                    cycle = cycle,
                    previous_state = previous_state,
                    motion_permitted = False,
                    permitted_target = None,
                    reasons = ("explicit_reenable_required",)
                )

            return self._result(
                cycle = cycle,
                previous_state = previous_state,
                motion_permitted = False,
                permitted_target = None,
                reasons = ("stability_recovering",)
            )

        raise RuntimeError(f"Unhandled supervisor state: {self._state}")

    def _validate_cycle_order(self, cycle: SupervisorCycleInput) -> None:
        if cycle.cycle_sequence_id < 0:
            raise ValueError("cycle_sequence_id must be non-negative.")

        if not math.isfinite(cycle.decision_timestamp_s):
            raise ValueError("decision_timestamp_s must be finite.")

        if cycle.decision_timestamp_s < 0.0:
            raise ValueError("decision_timestamp_s must be non-negative.")

        if self._last_cycle_sequence_id is not None and cycle.cycle_sequence_id <= self._last_cycle_sequence_id:
            raise ValueError("Supervisor cycle sequence IDs must be strictly increasing.")

        if self._last_decision_timestamp_s is not None and cycle.decision_timestamp_s <= self._last_decision_timestamp_s:
            raise ValueError("Supervisor decision timestamps must be strictly increasing.")

    def _fault_reasons(self, cycle: SupervisorCycleInput) -> tuple[str, ...]:
        reasons: list[str] = []

        if cycle.controller_fault:
            reasons.append("controller_fault")
        if cycle.persistent_inconsistency:
            reasons.append("persistent_inconsistency")
        if cycle.external_stop:
            reasons.append("external_stop")

        return tuple(reasons)

    def _required_validity_reasons(self, cycle: SupervisorCycleInput) -> tuple[str, ...]:
        base_reasons = self._base_validity_reasons(cycle)
        if self.config.gripper_loss_policy == GripperLossPolicy.HOLD_TELEOP:
            return _merge_reasons(base_reasons, self._gripper_validity_reasons(cycle))
        return base_reasons

    def _base_validity_reasons(self, cycle: SupervisorCycleInput) -> tuple[str, ...]:
        reasons: list[str] = []

        if not cycle.human_valid:
            reasons.append("human_tracking_invalid")

        if cycle.human_frame_sequence_id is None:
            reasons.append("human_frame_sequence_missing")
        elif cycle.human_frame_sequence_id < 0:
            reasons.append("human_frame_sequence_invalid")

        human_timestamp_ok = _append_age_reasons(
            reasons = reasons,
            label = "human",
            timestamp_s = cycle.human_measurement_timestamp_s,
            decision_timestamp_s = cycle.decision_timestamp_s,
            max_age_s = self.config.max_human_age_s
        )

        if not cycle.robot_state_valid:
            reasons.append("robot_state_invalid")

        robot_timestamp_ok = _append_age_reasons(
            reasons = reasons,
            label = "robot_state",
            timestamp_s = cycle.robot_state_timestamp_s,
            decision_timestamp_s = cycle.decision_timestamp_s,
            max_age_s = self.config.max_robot_state_age_s
        )

        if not cycle.transform_available:
            reasons.append("transform_unavailable")

        transform_timestamp_ok = True
        if cycle.transform_available and cycle.transform_timestamp_s is not None:
            transform_timestamp_ok = _append_age_reasons(
                reasons = reasons,
                label = "transform",
                timestamp_s = cycle.transform_timestamp_s,
                decision_timestamp_s = cycle.decision_timestamp_s,
                max_age_s = self.config.max_dynamic_transform_age_s
            )

        if not cycle.controller_ready:
            reasons.append("controller_not_ready")

        if human_timestamp_ok and robot_timestamp_ok:
            assert cycle.human_measurement_timestamp_s is not None
            assert cycle.robot_state_timestamp_s is not None
            if abs(cycle.human_measurement_timestamp_s - cycle.robot_state_timestamp_s) > self.config.max_human_robot_skew_s:
                reasons.append("human_robot_unsynchronized")

        if human_timestamp_ok and transform_timestamp_ok and cycle.transform_timestamp_s is not None:
            assert cycle.human_measurement_timestamp_s is not None
            if abs(cycle.human_measurement_timestamp_s - cycle.transform_timestamp_s) > self.config.max_human_transform_skew_s:
                reasons.append("human_transform_unsynchronized")

        return tuple(reasons)

    def _target_validity_reasons(self, cycle: SupervisorCycleInput) -> tuple[str, ...]:
        reasons: list[str] = []
        target = cycle.retargeting_result

        if target is None:
            return ("retargeting_missing",)
        if not target.active:
            reasons.append("retargeting_inactive")
        if not target.valid:
            reasons.append("retargeting_invalid")
        if target.target_pose is None:
            reasons.append("retargeting_target_missing")

        target_timestamp_ok = _append_age_reasons(
            reasons = reasons,
            label = "target",
            timestamp_s = target.measurement_timestamp_s,
            decision_timestamp_s = cycle.decision_timestamp_s,
            max_age_s = self.config.max_target_age_s
        )

        if (cycle.human_frame_sequence_id is not None and target.frame_sequence_id != cycle.human_frame_sequence_id):
            reasons.append("human_target_frame_mismatch")

        if (target_timestamp_ok and cycle.human_measurement_timestamp_s is not None and math.isfinite(cycle.human_measurement_timestamp_s)):
            if abs(target.measurement_timestamp_s - cycle.human_measurement_timestamp_s) > self.config.max_human_target_skew_s:
                reasons.append("human_target_unsynchronized")

        return tuple(reasons)

    def _gripper_validity_reasons(self, cycle: SupervisorCycleInput) -> tuple[str, ...]:
        if self.config.gripper_loss_policy is None:
            return ()

        reasons: list[str] = []
        gripper = cycle.gripper_aperture_result

        if gripper is None:
            return ("gripper_aperture_missing",)

        if not gripper.measurement_valid:
            reasons.append("gripper_aperture_invalid")
            for source_reason in gripper.reasons:
                reasons.append(f"gripper_source_{source_reason}")

        aperture = gripper.normalized_aperture
        if aperture is None:
            reasons.append("gripper_aperture_value_missing")
        elif not math.isfinite(aperture):
            reasons.append("gripper_aperture_value_invalid")
        elif aperture < 0.0 or aperture > 1.0:
            reasons.append("gripper_aperture_value_out_of_range")

        if gripper.frame_sequence_id < 0:
            reasons.append("gripper_frame_sequence_invalid")
        elif (cycle.human_frame_sequence_id is not None and gripper.frame_sequence_id != cycle.human_frame_sequence_id):
            reasons.append("human_gripper_frame_mismatch")

        if not math.isfinite(gripper.measurement_timestamp_s):
            reasons.append("gripper_timestamp_invalid")
        elif gripper.measurement_timestamp_s < 0.0:
            reasons.append("gripper_timestamp_invalid")
        elif (cycle.human_measurement_timestamp_s is not None and math.isfinite(cycle.human_measurement_timestamp_s)
            and gripper.measurement_timestamp_s != cycle.human_measurement_timestamp_s):
            reasons.append("human_gripper_timestamp_mismatch")

        return _deduplicate_reasons(reasons)

    def _gripper_output(self, cycle: SupervisorCycleInput) -> tuple[bool, float | None, tuple[str, ...]]:
        if self.config.gripper_loss_policy is None:
            return False, None, ()

        reasons = self._gripper_validity_reasons(cycle)
        if reasons:
            return False, None, reasons

        gripper = cycle.gripper_aperture_result
        assert gripper is not None
        assert gripper.normalized_aperture is not None

        return True, gripper.normalized_aperture, ()

    def _result(
        self,
        cycle: SupervisorCycleInput,
        previous_state: SupervisorState,
        motion_permitted: bool,
        permitted_target: RobotAgnosticPose | None,
        reasons: tuple[str, ...],
    ) -> CommandSupervisorResult:
        if motion_permitted:
            if self._state != SupervisorState.ACTIVE:
                raise RuntimeError("Motion can only be permitted in ACTIVE.")
            if permitted_target is None:
                raise RuntimeError("ACTIVE motion permission requires a target.")
        else:
            permitted_target = None

        gripper_command_permitted = False
        permitted_gripper_aperture: float | None = None
        gripper_reasons: tuple[str, ...] = ()

        if self._state == SupervisorState.ACTIVE and motion_permitted:
            (
                gripper_command_permitted,
                permitted_gripper_aperture,
                gripper_reasons
            ) = self._gripper_output(cycle)
        elif self.config.gripper_loss_policy is not None:
            gripper_reasons = self._gripper_validity_reasons(cycle)

        if not gripper_command_permitted:
            permitted_gripper_aperture = None

        return CommandSupervisorResult(
            cycle_sequence_id = cycle.cycle_sequence_id,
            decision_timestamp_s = cycle.decision_timestamp_s,
            previous_state = previous_state,
            state = self._state,
            state_changed = (previous_state != self._state),
            motion_permitted = motion_permitted,
            permitted_target = permitted_target,
            gripper_command_permitted = gripper_command_permitted,
            permitted_gripper_aperture = permitted_gripper_aperture,
            consecutive_valid_cycles = self._consecutive_valid_cycles,
            reasons = reasons,
            gripper_reasons = gripper_reasons
        )


def _append_age_reasons(reasons: list[str], label: str, timestamp_s: float | None, decision_timestamp_s: float, max_age_s: float) -> bool:
    if timestamp_s is None:
        reasons.append(f"{label}_timestamp_missing")
        return False

    if not math.isfinite(timestamp_s):
        reasons.append(f"{label}_timestamp_invalid")
        return False

    age_s = decision_timestamp_s - timestamp_s

    if age_s < 0.0:
        reasons.append(f"{label}_timestamp_in_future")
        return False
    if age_s > max_age_s:
        reasons.append(f"{label}_stale")
        return False
    return True


def _merge_reasons(first: tuple[str, ...], second: tuple[str, ...]) -> tuple[str, ...]:
    return _deduplicate_reasons([*first, *second])

def _deduplicate_reasons(reasons: list[str]) -> tuple[str, ...]:
    merged: list[str] = []

    for reason in reasons:
        if reason not in merged:
            merged.append(reason)

    return tuple(merged)