from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, replace
from types import SimpleNamespace
import numpy as np
from human_tracking.acquisition import CameraFrame
from human_tracking.mediapipe_adapter import MediaPipeObservationAdapter
from human_tracking.mediapipe_backend import MediaPipeTrackingOutput
from human_tracking.observation import (
    ControlArm,
    HandAssociationSource,
    HandObservation,
    HumanObservation,
    Landmark,
    LandmarkSet,
    ObservationValidationConfig,
    associate_hands_to_pose,
    evaluate_observation,
)
from human_tracking.tracking import (
    DerivedGeometryConfig,
    HumanLandmarkSmoother,
    HumanTemporalTracker,
    LandmarkSmoothingConfig,
    SmoothedChannelLandmarks,
    SmoothedHumanLandmarks,
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
from human_tracking.supervisor import (
    CommandSupervisor,
    CommandSupervisorConfig,
    GripperLossPolicy,
    SupervisorCycleInput,
    SupervisorState,
)
from human_tracking.gripper_intent import (
    GripperApertureConfig,
    GripperApertureTracker,
    GripperIntentConfig,
    GripperIntentState,
    GripperIntentTracker,
    HandSide,
)

def fake_landmarks(count: int, with_confidence: bool, visibility: float = 0.90, presence: float = 0.95):
    landmarks = []

    for index in range(count):
        landmark = SimpleNamespace(
            x = 0.01 * index,
            y = 0.02 * index,
            z = -0.001 * index,
        )

        if with_confidence:
            landmark.visibility = visibility
            landmark.presence = presence

        landmarks.append(landmark)

    return landmarks

def fake_handedness(name: str, score: float):
    return [SimpleNamespace(category_name=name, score=score)]

def make_frame():
    return CameraFrame(
        sequence_id = 7,
        measurement_timestamp_s = 123.456,
        image_bgr = np.zeros((480, 640, 3), dtype = np.uint8),
        image_width_px = 640,
        image_height_px = 480,
    )

def make_complete_output():
    return MediaPipeTrackingOutput(
        frame_sequence_id = 7,
        measurement_timestamp_s = 123.456,
        mediapipe_timestamp_ms = 123456,
        pose_result=SimpleNamespace(
            pose_landmarks = [fake_landmarks(33, with_confidence = True)],
            pose_world_landmarks=[fake_landmarks(33, with_confidence = True,)],
        ),
        hand_result = SimpleNamespace(
            hand_landmarks = [
                fake_landmarks(21, with_confidence=False,),
                fake_landmarks(21, with_confidence=False,),
            ],
            hand_world_landmarks = [
                fake_landmarks(21, with_confidence=False,),
                fake_landmarks(21, with_confidence=False,),
            ],
            handedness = [
                fake_handedness("Left", 0.97,),
                fake_handedness("Right", 0.96,),
            ],
        ),
        processing_duration_s=0.05,
    )

def shifted_observation(source, sequence_id: int, timestamp_s: float):
    return replace(source, frame_sequence_id = sequence_id, measurement_timestamp_s = timestamp_s)

def with_body_landmark_x(source, landmark_index: int, x: float, sequence_id: int, timestamp_s: float):
    body_set = source.body_image_landmarks

    if body_set is None:
        raise ValueError("Source observation has no body landmarks.")

    landmarks = list(body_set.landmarks)
    landmarks[landmark_index] = replace(landmarks[landmark_index],x = x)

    changed_body_set = LandmarkSet(
        coordinate_space = (body_set.coordinate_space),
        landmarks = tuple(landmarks)
    )

    return replace(
        source,
        frame_sequence_id = sequence_id,
        measurement_timestamp_s = timestamp_s,
        body_image_landmarks = changed_body_set,
    )

def with_body_landmark_confidence(
    source,
    landmark_index: int,
    visibility: float | None,
    presence: float | None,
):
    body_set = source.body_image_landmarks

    if body_set is None:
        raise ValueError("Source observation has no body landmarks.")

    landmarks = list(body_set.landmarks)
    landmarks[landmark_index] = replace(
        landmarks[landmark_index],
        visibility = visibility,
        presence = presence)

    changed_body_set = LandmarkSet(
        coordinate_space = body_set.coordinate_space,
        landmarks = tuple(landmarks))

    return replace(
        source,
        body_image_landmarks = changed_body_set)


def landmark_set_with_positions(source: LandmarkSet, positions: dict[int, tuple[float, float, float]]) -> LandmarkSet:
    landmarks = list(source.landmarks)

    for index, position in positions.items():
        current = landmarks[index]

        landmarks[index] = Landmark(
            index=current.index,
            name=current.name,
            x=position[0],
            y=position[1],
            z=position[2],
            visibility=current.visibility,
            presence=current.presence)

    return LandmarkSet(
        coordinate_space = (source.coordinate_space),
        landmarks=tuple(landmarks))


def hand_with_wrist(
    hand: HandObservation,
    *,
    x: float,
    y: float,
    handedness: str | None = None,
    handedness_score: float | None = None,
) -> HandObservation:
    image_landmarks = landmark_set_with_positions(
        hand.image_landmarks,
        {0: (x, y, 0.0)},
    )
    return replace(
        hand,
        handedness = (hand.handedness if handedness is None else handedness),
        handedness_score = (hand.handedness_score if handedness_score is None else handedness_score),
        image_landmarks = image_landmarks)

def observation_with_pose_anchors(source: HumanObservation) -> HumanObservation:
    body = source.body_image_landmarks
    if body is None:
        raise ValueError("Source observation has no body image landmarks.")

    return replace(
        source,
        body_image_landmarks = landmark_set_with_positions(
            body,
            {
                11: (0.30, 0.40, 0.0),
                12: (0.70, 0.40, 0.0),
                15: (0.20, 0.65, 0.0),
                16: (0.80, 0.65, 0.0),
            },
        ),
    )


def assert_close(
    actual: float,
    expected: float,
    tolerance: float = 1e-9,
):
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"Expected {expected}, received {actual}.")

@contextmanager
def _expect_value_error(message: str):
    try:
        yield
    except ValueError:
        return
    raise AssertionError(message)

@dataclass(frozen=True)
class _ObservationFixtures:
    observation: HumanObservation
    missing_observation: HumanObservation
    config: ObservationValidationConfig
    temporal_config: TemporalTrackingConfig

def _run_h3_h4_checks() -> _ObservationFixtures:
    frame = make_frame()
    adapter = MediaPipeObservationAdapter()

    observation = adapter.convert(frame, make_complete_output())

    assert observation.frame_sequence_id == 7
    assert observation.measurement_timestamp_s == 123.456
    assert observation.image_width_px == 640
    assert observation.image_height_px == 480

    assert observation.body_image_landmarks is not None
    assert len(observation.body_image_landmarks) == 33

    assert observation.body_image_landmarks.landmarks[11].name == "left_shoulder"

    assert len(observation.left_hands) == 1
    assert len(observation.right_hands) == 1
    assert len(observation.unknown_hands) == 0

    assert len(observation.left_hands[0].image_landmarks) == 21

    assert observation.left_hands[0].image_landmarks.landmarks[0].name == "wrist"

    serialized = observation.to_json()

    assert '"measurement_timestamp_s":123.456' in serialized

    assert '"coordinate_space":"normalized_image"' in serialized

    print("H3 complete observation: PASS")
    print("H3 timestamp preservation: PASS")
    print("H3 body schema: PASS")
    print("H3 hand schema: PASS")
    print("H3 left/right separation: PASS")
    print("H3 JSON serialization: PASS")

    missing_output = MediaPipeTrackingOutput(
        frame_sequence_id = 7,
        measurement_timestamp_s = 123.456,
        mediapipe_timestamp_ms = 123456,
        pose_result = SimpleNamespace(
            pose_landmarks = [],
            pose_world_landmarks = [],
        ),
        hand_result = SimpleNamespace(
            hand_landmarks = [],
            hand_world_landmarks = [],
            handedness = [],
        ),
        processing_duration_s = 0.05,
    )

    missing_observation = adapter.convert(frame, missing_output)

    assert missing_observation.body_image_landmarks is None
    assert missing_observation.body_world_landmarks is None
    assert not missing_observation.left_hands
    assert not missing_observation.right_hands
    assert not missing_observation.unknown_hands

    print("H3 explicit missing data: PASS")

    config = ObservationValidationConfig(
        selected_control_arm = ControlArm.RIGHT,
        min_body_visibility = 0.50,
        min_body_presence = 0.50,
        min_handedness_score = 0.50,
    )

    left_arm_config = ObservationValidationConfig(
        selected_control_arm = ControlArm.LEFT,
        min_body_visibility = 0.50,
        min_body_presence = 0.50,
        min_handedness_score = 0.50,
    )

    valid = evaluate_observation(observation, config)

    assert valid.available_for_visualization
    assert valid.body_valid_for_control
    assert valid.left_hand_valid_for_control
    assert valid.right_hand_valid_for_control
    assert not valid.body_reasons
    assert not valid.left_hand_reasons
    assert not valid.right_hand_reasons

    print("H4 complete valid frame: PASS")

    right_selected_unused_left_invalid = with_body_landmark_confidence(observation, landmark_index = 13, visibility = 0.10, presence = 0.10)
    right_selected_unused_left_invalid = with_body_landmark_confidence(right_selected_unused_left_invalid, landmark_index = 15, visibility = None, presence = None)
    right_selected_validity = evaluate_observation(right_selected_unused_left_invalid, config)

    assert right_selected_validity.body_valid_for_control
    assert not right_selected_validity.body_reasons

    left_selected_unused_right_invalid = with_body_landmark_confidence(observation, landmark_index = 14, visibility = 0.10, presence = 0.10)
    left_selected_unused_right_invalid = with_body_landmark_confidence(left_selected_unused_right_invalid, landmark_index = 16, visibility = None, presence = None)
    left_selected_validity = evaluate_observation(left_selected_unused_right_invalid, left_arm_config)

    assert left_selected_validity.body_valid_for_control
    assert not left_selected_validity.body_reasons

    print("P4A unused-arm confidence isolation: PASS")

    selected_arm_required_cases = (
        (config, 14, "right_elbow_visibility_low"),
        (config, 16, "right_wrist_visibility_low"),
        (left_arm_config, 13, "left_elbow_visibility_low"),
        (left_arm_config, 15, "left_wrist_visibility_low")
    )

    for selected_config, index, expected_reason in (selected_arm_required_cases):
        invalid_selected_arm = (
            with_body_landmark_confidence(
                observation,
                landmark_index = index,
                visibility = 0.10,
                presence = 0.95,
            )
        )

        invalid_selected_arm_validity = evaluate_observation(invalid_selected_arm, selected_config)
        assert not invalid_selected_arm_validity.body_valid_for_control
        assert expected_reason in invalid_selected_arm_validity.body_reasons

    for index in (11, 12, 23, 24):
        invalid_torso = with_body_landmark_confidence(
            observation,
            landmark_index = index,
            visibility = 0.90,
            presence = 0.10,
        )

        invalid_torso_validity = evaluate_observation(invalid_torso, config)
        assert not invalid_torso_validity.body_valid_for_control

    print("P4A selected-arm/torso required landmarks: PASS")

    missing_validity = evaluate_observation(missing_observation, config)
    assert not missing_validity.available_for_visualization
    assert not missing_validity.body_valid_for_control
    assert not missing_validity.left_hand_valid_for_control
    assert not missing_validity.right_hand_valid_for_control
    assert "body_missing" in missing_validity.body_reasons
    assert "left_hand_missing" in missing_validity.left_hand_reasons
    assert "right_hand_missing" in missing_validity.right_hand_reasons
    print("H4 missing-data rejection: PASS")

    low_confidence_output = MediaPipeTrackingOutput(
        frame_sequence_id = 7,
        measurement_timestamp_s = 123.456,
        mediapipe_timestamp_ms = 123456,
        pose_result=SimpleNamespace(
            pose_landmarks = [
                fake_landmarks(33,
                    with_confidence = True,
                    visibility = 0.20,
                    presence = 0.20,
                )
            ],
            pose_world_landmarks = [
                fake_landmarks(33,
                    with_confidence = True,
                    visibility = 0.20,
                    presence = 0.20,
                )
            ],
        ),
        hand_result = (make_complete_output().hand_result),
        processing_duration_s = 0.05,
    )

    low_confidence_observation = adapter.convert(frame, low_confidence_output)
    low_confidence_validity = evaluate_observation(low_confidence_observation, config)

    assert low_confidence_validity.available_for_visualization
    assert not low_confidence_validity.body_valid_for_control
    assert low_confidence_validity.left_hand_valid_for_control
    assert low_confidence_validity.right_hand_valid_for_control
    assert any(reason.endswith("_visibility_low") for reason in low_confidence_validity.body_reasons)
    assert any(reason.endswith("_presence_low") for reason in low_confidence_validity.body_reasons)

    print("H4 visualization/control separation: PASS")

    low_hand_output = MediaPipeTrackingOutput(
        frame_sequence_id = 7,
        measurement_timestamp_s = 123.456,
        mediapipe_timestamp_ms = 123456,
        pose_result = (make_complete_output().pose_result),
        hand_result = SimpleNamespace(
            hand_landmarks = [
                fake_landmarks(21, with_confidence = False),
                fake_landmarks(21, with_confidence = False),
            ],
            hand_world_landmarks = [
                fake_landmarks(21, with_confidence=False),
                fake_landmarks(21, with_confidence=False),
            ],
            handedness=[
                fake_handedness("Left", 0.25),
                fake_handedness("Right", 0.96),
            ],
        ),
        processing_duration_s = 0.05,
    )

    low_hand_observation = adapter.convert(frame, low_hand_output)
    low_hand_validity = evaluate_observation(low_hand_observation, config)

    assert not low_hand_validity.left_hand_valid_for_control
    assert low_hand_validity.right_hand_valid_for_control
    assert "left_handedness_score_low" in low_hand_validity.left_hand_reasons

    print("H4 handedness-confidence rejection: PASS")

    with _expect_value_error('Invalid threshold was accepted.'):
        ObservationValidationConfig(
            selected_control_arm = ControlArm.RIGHT,
            min_body_visibility = 1.5,
        )

    print("H4 threshold configuration check: PASS")

    with _expect_value_error('Invalid selected control arm was accepted.'):
        ObservationValidationConfig( selected_control_arm = "center")

    print("P4A control-arm configuration guard: PASS")

    temporal_config = TemporalTrackingConfig(
        consecutive_valid_required = 3,
        dropout_timeout_s = 0.25,
    )

    tracker = HumanTemporalTracker(temporal_config)

    temporal_1 = tracker.update(
        shifted_observation(
            observation,
            sequence_id = 1,
            timestamp_s = 10.000,
        ),
        valid,
    )

    assert temporal_1.body.state == TrackingState.ACQUIRING
    assert not temporal_1.body.valid_for_control

    temporal_2 = tracker.update(
        shifted_observation(
            observation,
            sequence_id = 2,
            timestamp_s = 10.067,
        ),
        valid,
    )

    assert temporal_2.body.state == TrackingState.ACQUIRING
    assert not temporal_2.body.valid_for_control

    temporal_3 = tracker.update(
        shifted_observation(
            observation,
            sequence_id = 3,
            timestamp_s = 10.134,
        ),
        valid,
    )

    assert temporal_3.body.state == TrackingState.TRACKING
    assert temporal_3.body.valid_for_control
    assert temporal_3.left_hand.valid_for_control
    assert temporal_3.right_hand.valid_for_control

    print("H4 temporal initial acquisition gate: PASS")

    temporal_4 = tracker.update(
        shifted_observation(
            missing_observation,
            sequence_id = 4,
            timestamp_s = 10.201,
        ),
        missing_validity,
    )

    assert temporal_4.body.state == TrackingState.DROPOUT
    assert not temporal_4.body.valid_for_control
    assert temporal_4.body.age_since_last_valid_s is not None
    assert temporal_4.body.age_since_last_valid_s < temporal_config.dropout_timeout_s

    print("H4 temporal short-dropout handling: PASS")

    temporal_5 = tracker.update(
        shifted_observation(
            missing_observation,
            sequence_id = 5,
            timestamp_s = 10.500,
        ),
        missing_validity,
    )

    assert temporal_5.body.state == TrackingState.LOST
    assert not temporal_5.body.valid_for_control

    print("H4 temporal loss-timeout handling: PASS")

    temporal_6 = tracker.update(
        shifted_observation(
            observation,
            sequence_id = 6,
            timestamp_s = 10.567,
        ),
        valid,
    )

    assert temporal_6.body.state == TrackingState.ACQUIRING
    assert not temporal_6.body.valid_for_control

    temporal_7 = tracker.update(
        shifted_observation(
            observation,
            sequence_id = 7,
            timestamp_s = 10.634,
        ),
        valid,
    )

    assert temporal_7.body.state == TrackingState.ACQUIRING
    assert not temporal_7.body.valid_for_control

    temporal_8 = tracker.update(
        shifted_observation(
            observation,
            sequence_id = 8,
            timestamp_s = 10.701,
        ),
        valid,
    )

    assert temporal_8.body.state == TrackingState.TRACKING
    assert temporal_8.body.valid_for_control

    print("H4 temporal reacquisition gate: PASS")

    with _expect_value_error('Duplicate sequence ID was accepted.'):
        tracker.update(
            shifted_observation(
                observation,
                sequence_id = 8,
                timestamp_s = 10.768,
            ),
            valid,
        )

    print("H4 temporal ordering guard: PASS")

    return _ObservationFixtures(
        observation = observation,
        missing_observation = missing_observation,
        config = config,
        temporal_config = temporal_config,
    )

def _run_ht1_checks(fixtures: _ObservationFixtures) -> None:
    base = observation_with_pose_anchors(fixtures.observation)
    config = ObservationValidationConfig(
        selected_control_arm = ControlArm.LEFT,
        min_body_visibility = 0.50,
        min_body_presence = 0.50,
        min_handedness_score = 0.50,
        enable_pose_hand_association = True,
        max_hand_wrist_pose_distance_shoulder_widths = 0.50,
    )

    misclassified_left = hand_with_wrist(
        base.left_hands[0],
        x = 0.205,
        y = 0.648,
        handedness = "Right",
        handedness_score = 0.99,
    )
    mislabeled_observation = replace(
        base,
        left_hands = (),
        right_hands = (misclassified_left,),
        unknown_hands = (),
    )
    associated = associate_hands_to_pose(mislabeled_observation, config)

    assert len(associated.left_hands) == 1
    assert not associated.right_hands
    assert associated.left_hands[0].handedness == "Right"
    assert associated.left_hands[0].handedness_score == 0.99
    assert associated.left_hands[0].association_source == HandAssociationSource.POSE_WRIST
    associated_validity = evaluate_observation(associated, config)
    assert associated_validity.left_hand_valid_for_control
    print("HT1 misclassified selected-hand pose association: PASS")


    left_candidate = hand_with_wrist(
        base.left_hands[0],
        x = 0.208,
        y = 0.652,
        handedness = "Left",
        handedness_score = 0.95,
    )
    right_candidate = hand_with_wrist(
        base.right_hands[0],
        x = 0.792,
        y = 0.648,
        handedness = "Left",
        handedness_score = 0.94,
    )
    ambiguous_backend = replace(
        base,
        left_hands = (left_candidate, right_candidate),
        right_hands = (),
        unknown_hands = (),
    )
    before = evaluate_observation(ambiguous_backend, config)
    assert "left_hand_ambiguous" in before.left_hand_reasons

    split = associate_hands_to_pose(ambiguous_backend, config)
    assert len(split.left_hands) == 1
    assert len(split.right_hands) == 1
    split_validity = evaluate_observation(split, config)
    assert split_validity.left_hand_valid_for_control
    assert split_validity.right_hand_valid_for_control
    assert split.right_hands[0].handedness == "Left"
    print("HT1 same-label two-hand disambiguation: PASS")

    farther_left = hand_with_wrist(
        base.right_hands[0],
        x = 0.25,
        y = 0.65,
        handedness = "Right",
        handedness_score = 0.91,
    )
    duplicate_left = replace(
        base,
        left_hands = (left_candidate,),
        right_hands = (farther_left,),
        unknown_hands = (),
    )

    deduplicated = associate_hands_to_pose(duplicate_left, config)
    assert len(deduplicated.left_hands) == 1
    assert deduplicated.left_hands[0].handedness == "Left"
    assert not deduplicated.right_hands
    assert len(deduplicated.unknown_hands) == 1
    print("HT1 nearest-candidate duplicate suppression: PASS")

    far_hand = hand_with_wrist(
        base.right_hands[0],
        x = 0.50,
        y = 0.95,
        handedness = "Right",
        handedness_score = 0.99,
    )
    far_observation = replace(
        base,
        left_hands = (),
        right_hands = (far_hand,),
        unknown_hands = (),
    )
    far_associated = associate_hands_to_pose(far_observation, config)
    assert not far_associated.left_hands
    assert not far_associated.right_hands
    assert len(far_associated.unknown_hands) == 1
    assert "left_hand_missing" in evaluate_observation(far_associated, config).left_hand_reasons
    print("HT1 shoulder-width association gate: PASS")

    body = base.body_image_landmarks
    assert body is not None
    right_wrist_low = list(body.landmarks)
    right_wrist_low[16] = replace(
        right_wrist_low[16],
        visibility = 0.10,
        presence = 0.10,
    )
    one_wrist_pose = replace(
        mislabeled_observation,
        body_image_landmarks = LandmarkSet(
            coordinate_space = body.coordinate_space,
            landmarks=tuple(right_wrist_low),
        ),
    )
    one_wrist_associated = associate_hands_to_pose(one_wrist_pose, config)
    assert len(one_wrist_associated.left_hands) == 1
    assert one_wrist_associated.left_hands[0].association_source == HandAssociationSource.POSE_WRIST
    print("HT1 unused-pose-wrist confidence isolation: PASS")

    low_anchor = with_body_landmark_confidence(
        mislabeled_observation,
        landmark_index = 11,
        visibility = 0.10,
        presence = 0.10,
    )
    fallback = associate_hands_to_pose(low_anchor, config)
    assert fallback == low_anchor
    print("HT1 pose-anchor fail-closed fallback: PASS")

    serialized = associated.to_dict()
    assert serialized["left_hands"][0]["association_source"] == "pose_wrist"
    print("HT1 association-source serialization: PASS")

    with _expect_value_error("Non-positive hand-association gate was accepted."):
        ObservationValidationConfig(
            selected_control_arm = ControlArm.LEFT,
            enable_pose_hand_association = True,
            max_hand_wrist_pose_distance_shoulder_widths = 0.0,
        )
    with _expect_value_error("Non-finite hand-association gate was accepted."):
        ObservationValidationConfig(
            selected_control_arm = ControlArm.LEFT,
            enable_pose_hand_association = True,
            max_hand_wrist_pose_distance_shoulder_widths = float("nan"),
        )
    print("HT1 hand-association configuration guards: PASS")

def _run_h5_checks(fixtures: _ObservationFixtures) -> None:
    observation = fixtures.observation
    missing_observation = fixtures.missing_observation
    config = fixtures.config
    temporal_config = fixtures.temporal_config

    # ---------------------------------------------------------
    # H5 smoothing tests
    # ---------------------------------------------------------

    smoothing_config = LandmarkSmoothingConfig(time_constant_s = 0.10)
    smoothing_tracker = HumanTemporalTracker(temporal_config)
    smoother = HumanLandmarkSmoother(smoothing_config)

    # first 2 frames do not require smoothing
    smooth_obs_1 = with_body_landmark_x(
        observation,
        landmark_index = 15,
        x = 0.50,
        sequence_id = 1,
        timestamp_s = 20.000,
    )

    smooth_valid_1 = evaluate_observation(smooth_obs_1, config)
    smooth_track_1 = smoothing_tracker.update(smooth_obs_1, smooth_valid_1)
    smooth_result_1 = smoother.update(smooth_obs_1, smooth_track_1)

    assert smooth_result_1.body.image_landmarks is None

    smooth_obs_2 = with_body_landmark_x(
        observation,
        landmark_index = 15,
        x = 0.55,
        sequence_id = 2,
        timestamp_s = 20.067,
    )

    smooth_valid_2 = evaluate_observation(smooth_obs_2, config)
    smooth_track_2 = smoothing_tracker.update(smooth_obs_2, smooth_valid_2)
    smooth_result_2 = smoother.update(smooth_obs_2, smooth_track_2)

    assert smooth_result_2.body.image_landmarks is None

    # Frame 3 reaches TRACKING and seeds the filter.
    smooth_obs_3 = with_body_landmark_x(
        observation,
        landmark_index=15,
        x=0.50,
        sequence_id=3,
        timestamp_s=20.134,
    )

    smooth_valid_3 = evaluate_observation(smooth_obs_3, config)
    smooth_track_3 = smoothing_tracker.update(smooth_obs_3, smooth_valid_3)
    smooth_result_3 = smoother.update(smooth_obs_3, smooth_track_3)

    assert smooth_result_3.body.image_landmarks is not None

    seed_x = smooth_result_3.body.image_landmarks.landmarks[15].x

    assert abs(seed_x - 0.50) < 1e-12

    # A sudden 0.10 raw jump must be reduced by the low-pass filter.
    smooth_obs_4 = with_body_landmark_x(
        observation,
        landmark_index=15,
        x = 0.60,
        sequence_id = 4,
        timestamp_s = 20.201,
    )

    smooth_valid_4 = evaluate_observation(smooth_obs_4, config)
    smooth_track_4 = smoothing_tracker.update(smooth_obs_4, smooth_valid_4)
    smooth_result_4 = smoother.update(smooth_obs_4, smooth_track_4)

    assert smooth_result_4.body.image_landmarks is not None
    filtered_x = smooth_result_4.body.image_landmarks.landmarks[15].x
    assert 0.50 < filtered_x < 0.60
    raw_step = abs(0.60 - 0.50)
    filtered_step = abs(filtered_x - 0.50)
    assert filtered_step < raw_step

    print("H5 valid-measurement smoothing: PASS")

    # A dropout must emit no control-side smoothed body landmarks and reset the filter state.
    smooth_missing_5 = shifted_observation(
        missing_observation,
        sequence_id = 5,
        timestamp_s = 20.268,
    )

    smooth_missing_validity_5 = evaluate_observation(smooth_missing_5, config)
    smooth_track_5 = smoothing_tracker.update(smooth_missing_5, smooth_missing_validity_5)
    smooth_result_5 = smoother.update(smooth_missing_5,smooth_track_5)

    assert smooth_result_5.body.image_landmarks is None

    print("H5 invalid-measurement suppression: PASS")

    # Reacquisition again requires three frames. The filter is reset, so the first control-valid result seeds directly from fresh data.
    for sequence_id, timestamp_s in ((6, 20.335), (7, 20.402), (8, 20.469)):
        reacquired_observation = (
            with_body_landmark_x(
                observation,
                landmark_index = 15,
                x = 0.80,
                sequence_id = sequence_id,
                timestamp_s = timestamp_s,
            )
        )

        reacquired_validity = evaluate_observation(reacquired_observation, config)
        reacquired_tracking = smoothing_tracker.update(reacquired_observation, reacquired_validity)
        reacquired_smoothed = smoother.update(reacquired_observation, reacquired_tracking)

    assert reacquired_tracking.body.state == TrackingState.TRACKING
    assert reacquired_smoothed.body.image_landmarks is not None
    reacquired_x = reacquired_smoothed.body.image_landmarks.landmarks[15].x
    assert abs(reacquired_x - 0.80) < 1e-12
    print("H5 reset-on-dropout/reacquisition: PASS")

    with _expect_value_error('Invalid smoothing time constant was accepted.'):
        LandmarkSmoothingConfig(time_constant_s = 0.0)

    print("H5 smoothing configuration check: PASS")

def _run_h6_checks(observation: HumanObservation):
    if observation.body_world_landmarks is None:
        raise AssertionError("Synthetic observation has no body world landmarks.")

    if observation.left_hands[0].world_landmarks is None or observation.right_hands[0].world_landmarks is None:
        raise AssertionError("Synthetic observation has no hand world landmarks.")

    body_world = landmark_set_with_positions(
        observation.body_world_landmarks,
        {
            11: (-0.20, 0.40, 0.00),
            12: (0.20, 0.40, 0.00),
            13: (-0.40, 0.20, 0.00),
            14: (0.40, 0.20, 0.00),
            15: (-0.50, 0.00, 0.00),
            16: (0.50, 0.00, 0.00),
            23: (-0.15, 0.00, 0.00),
            24: (0.15, 0.00, 0.00),
        },
    )

    hand_positions = {
        0: (0.00, 0.00, 0.00),
        4: (0.01, 0.10, 0.00),
        5: (0.04, 0.04, 0.00),
        8: (0.02, 0.10, 0.00),
        9: (0.00, 0.08, 0.00),
        17: (-0.04, 0.04, 0.00),
    }

    left_hand_world = landmark_set_with_positions(observation.left_hands[0].world_landmarks, hand_positions)
    right_hand_world = landmark_set_with_positions(observation.right_hands[0].world_landmarks, hand_positions)
    left_selected_body_world = landmark_set_with_positions(
        body_world,
        {
            14: (float("nan"), float("nan"), float("nan")),
            16: (float("nan"), float("nan"), float("nan")),
        },
    )

    derived_input = SmoothedHumanLandmarks(
        frame_sequence_id = 42,
        measurement_timestamp_s = 30.000,
        body = SmoothedChannelLandmarks(
            image_landmarks = None,
            world_landmarks = left_selected_body_world,
        ),
        left_hand = SmoothedChannelLandmarks(
            image_landmarks = None,
            world_landmarks = left_hand_world,
        ),
        right_hand = SmoothedChannelLandmarks(
            image_landmarks = None,
            world_landmarks = right_hand_world,
        ),
    )

    derived = derive_human_kinematics(
        derived_input,
        DerivedGeometryConfig(
            selected_control_arm = ControlArm.LEFT,
            min_length_model_world = 1e-5,
        ),
    )

    assert derived.body is not None
    assert derived.left_hand is not None
    assert derived.right_hand is not None

    assert not derived.body_reasons
    assert not derived.left_hand_reasons
    assert not derived.right_hand_reasons

    assert_close(derived.body.frame.right_axis_model_world.x, 1.0)
    assert_close(derived.body.frame.up_axis_model_world.y, 1.0)
    assert_close(derived.body.frame.normal_axis_model_world.z, 1.0)
    assert_close(derived.body.shoulder_width_model_world, 0.40)
    assert_close(derived.body.torso_height_model_world, 0.40)

    left_wrist_displacement = derived.body.left_wrist_displacement_normalized_body
    assert left_wrist_displacement is not None
    assert left_wrist_displacement.x < 0.0
    assert left_wrist_displacement.y < 0.0

    assert derived.body.left_arm_length_model_world is not None
    assert derived.body.left_upper_arm_direction_body is not None
    assert derived.body.left_forearm_direction_body is not None

    assert derived.body.right_arm_length_model_world is None
    assert derived.body.right_upper_arm_direction_body is None
    assert derived.body.right_forearm_direction_body is None
    assert derived.body.right_wrist_displacement_normalized_body is None

    print("H6 body-relative frame and arm geometry: PASS")


    right_selected_body_world = landmark_set_with_positions(
        body_world,
        {
            13: (float("nan"), float("nan"), float("nan")),
            15: (float("nan"), float("nan"), float("nan")),
        },
    )

    right_derived_input = replace(
        derived_input,
        body = SmoothedChannelLandmarks(
            image_landmarks = None,
            world_landmarks = right_selected_body_world,
        ),
    )

    right_derived = derive_human_kinematics(
        right_derived_input,
        DerivedGeometryConfig(
            selected_control_arm = ControlArm.RIGHT,
            min_length_model_world = 1e-5,
        ),
    )

    assert right_derived.body is not None
    assert not right_derived.body_reasons

    right_wrist_displacement = right_derived.body.right_wrist_displacement_normalized_body

    assert right_wrist_displacement is not None
    assert right_wrist_displacement.x > 0.0
    assert right_wrist_displacement.y < 0.0

    assert right_derived.body.right_arm_length_model_world is not None
    assert right_derived.body.right_upper_arm_direction_body is not None
    assert right_derived.body.right_forearm_direction_body is not None

    assert right_derived.body.left_arm_length_model_world is None
    assert right_derived.body.left_upper_arm_direction_body is None
    assert right_derived.body.left_forearm_direction_body is None
    assert right_derived.body.left_wrist_displacement_normalized_body is None

    print("P4A selected-arm H6 geometry isolation: PASS")

    assert_close(derived.left_hand.palm_width_model_world, 0.08)
    assert_close(derived.left_hand.palm_length_model_world, 0.08)
    assert_close(derived.left_hand.pinch_ratio, 0.125)
    assert derived.left_hand.palm_normal_model_world.z > 0.99
    assert_close(derived.right_hand.pinch_ratio, 0.125)

    print("H6 normalized hand geometry: PASS")

    ht4_reference_hand_world = landmark_set_with_positions(
        left_hand_world,
        {
            0: (0.00, 0.00, 0.00),
            4: (0.02, 0.10, 0.00),
            5: (0.04, 0.04, 0.00),
            8: (0.06, 0.10, 0.00),
            9: (0.00, 0.08, 0.00),
            17: (-0.04, 0.04, 0.00),
        },
    )

    ht4_collapsed_width_hand_world = landmark_set_with_positions(
        left_hand_world,
        {
            0: (0.00, 0.00, 0.00),
            4: (0.02, 0.10, 0.00),
            5: (0.01, 0.04, 0.00),
            8: (0.06, 0.10, 0.00),
            9: (0.00, 0.08, 0.00),
            17: (-0.01, 0.04, 0.00),
        },
    )

    ht4_config = DerivedGeometryConfig(
        selected_control_arm = ControlArm.LEFT,
        min_length_model_world = 1e-5,
    )

    ht4_reference = derive_human_kinematics(
        replace(
            derived_input,
            left_hand = SmoothedChannelLandmarks(
                image_landmarks = None,
                world_landmarks = ht4_reference_hand_world,
            ),
            right_hand=SmoothedChannelLandmarks(
                image_landmarks = None,
                world_landmarks = None,
            ),
        ),
        ht4_config,
    )

    ht4_collapsed_width = derive_human_kinematics(
        replace(
            derived_input,
            left_hand = SmoothedChannelLandmarks(
                image_landmarks = None,
                world_landmarks = ht4_collapsed_width_hand_world,
            ),
            right_hand = SmoothedChannelLandmarks(
                image_landmarks = None,
                world_landmarks = None,
            ),
        ),
        ht4_config,
    )

    assert ht4_reference.left_hand is not None
    assert ht4_collapsed_width.left_hand is not None
    assert_close(ht4_reference.left_hand.palm_length_model_world, 0.08)
    assert_close(ht4_collapsed_width.left_hand.palm_length_model_world, 0.08)
    assert_close(ht4_reference.left_hand.pinch_ratio, 0.50)
    assert_close(ht4_collapsed_width.left_hand.pinch_ratio, 0.50)
    assert ht4_collapsed_width.left_hand.palm_width_model_world < ht4_reference.left_hand.palm_width_model_world

    print("HT4 palm-length-normalized pinch robustness: PASS")

    plausible_hand_world = landmark_set_with_positions(
        left_hand_world,
        {
            0: (0.00, 0.00, 0.00),
            4: (0.01, 0.10, 0.00),
            5: (0.03, 0.04, 0.00),
            8: (0.02, 0.10, 0.00),
            9: (0.00, 0.08, 0.00),
            17: (-0.03, 0.04, 0.00),
        },
    )

    implausible_hand_world = landmark_set_with_positions(
        left_hand_world,
        {
            0: (0.00, 0.00, 0.00),
            4: (0.04, 0.10, 0.00),
            5: (0.01, 0.04, 0.00),
            8: (-0.04, 0.10, 0.00),
            9: (0.00, 0.10, 0.00),
            17: (-0.01, 0.04, 0.00),
        },
    )

    plausibility_config = DerivedGeometryConfig(
        selected_control_arm = ControlArm.LEFT,
        min_length_model_world = 1e-5,
        min_palm_width_to_palm_length_ratio = 0.42,
    )

    plausible_geometry = derive_human_kinematics(
        replace(
            derived_input,
            left_hand = SmoothedChannelLandmarks(
                image_landmarks = None,
                world_landmarks = plausible_hand_world,
            ),
            right_hand = SmoothedChannelLandmarks(
                image_landmarks = None,
                world_landmarks = None,
            ),
        ),
        plausibility_config,
    )

    assert plausible_geometry.body is not None
    assert plausible_geometry.left_hand is not None
    assert not plausible_geometry.left_hand_reasons

    implausible_geometry = derive_human_kinematics(
        replace(
            derived_input,
            left_hand = SmoothedChannelLandmarks(
                image_landmarks = None,
                world_landmarks = implausible_hand_world,
            ),
            right_hand = SmoothedChannelLandmarks(
                image_landmarks = None,
                world_landmarks = None,
            ),
        ),
        plausibility_config,
    )

    assert implausible_geometry.body is not None
    assert implausible_geometry.left_hand is None
    assert "left_hand_palm_shape_implausible" in implausible_geometry.left_hand_reasons

    print("HT3 palm-shape plausibility rejection: PASS")
    print("HT3 body/arm isolation from bad hand geometry: PASS")

    with _expect_value_error("Invalid HT3 palm-shape ratio guard was accepted."):
        DerivedGeometryConfig(
            selected_control_arm = ControlArm.LEFT,
            min_palm_width_to_palm_length_ratio = 0.0,
        )

    with _expect_value_error("Non-finite HT3 palm-shape ratio guard was accepted."):
        DerivedGeometryConfig(
            selected_control_arm = ControlArm.LEFT,
            min_palm_width_to_palm_length_ratio = float("nan"),
        )

    print("HT3 palm-shape configuration guards: PASS")

    degenerate_body_world = landmark_set_with_positions(body_world, {12: (-0.20, 0.40, 0.00)})

    degenerate_input = SmoothedHumanLandmarks(
        frame_sequence_id = 43,
        measurement_timestamp_s = 30.067,
        body = SmoothedChannelLandmarks(
            image_landmarks = None,
            world_landmarks = degenerate_body_world,
        ),
        left_hand = SmoothedChannelLandmarks(
            image_landmarks=None,
            world_landmarks=None,
        ),
        right_hand = SmoothedChannelLandmarks(
            image_landmarks = None,
            world_landmarks = None,
        ),
    )

    degenerate = derive_human_kinematics(
        degenerate_input,
        DerivedGeometryConfig(selected_control_arm = ControlArm.LEFT)
    )

    assert degenerate.body is None
    assert ("body_shoulder_width_degenerate" in degenerate.body_reasons)
    assert degenerate.left_hand is None
    assert degenerate.right_hand is None

    print("H6 degenerate/missing geometry rejection: PASS")

    with _expect_value_error('Invalid derived-geometry guard was accepted.'):
        DerivedGeometryConfig(
            selected_control_arm = ControlArm.LEFT,
            min_length_model_world = 0.0,
        )

    print("H6 geometry configuration check: PASS")

    with _expect_value_error('Invalid H6 selected control arm was accepted.'):
        DerivedGeometryConfig(selected_control_arm = "center")

    print("P4A H6 control-arm configuration guard: PASS")
    return derived, right_derived


def _run_h8_checks(derived, right_derived):
    if derived.body is None:
        raise AssertionError("H8 requires the valid H6 synthetic body geometry.")

    reference_robot_pose = RobotAgnosticPose(
        position = Vector3(x = 1.0, y = 2.0, z = 3.0),
        orientation_xyzw=Quaternion(x = 0.0, y = 0.0, z = 0.0, w = 2.0),
    )

    mapping_config = RetargetingConfig(
        axis_mapping = AxisMapping(
            rows = (
                (0.0, 1.0, 0.0),
                (-1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        ),
        scale_robot_per_normalized_body = Vector3(x = 2.0, y = 3.0, z = 4.0),
        deadband_normalized_body = Vector3(x = 0.05, y = 0.05, z = 0.05),
        workspace_bounds = CartesianBounds(
            minimum = Vector3(x = -10.0, y = -10.0, z = -10.0),
            maximum = Vector3(x = 10.0, y = 10.0, z = 10.0),
        ),
        max_cartesian_speed_robot_per_s = None
    )

    retargeter = RelativeRetargeter(
        arm_side = ArmSide.LEFT,
        config = mapping_config,
    )

    reference = retargeter.activate(derived, reference_robot_pose)
    assert reference.arm_side == ArmSide.LEFT
    assert_close(reference.robot_pose_reference.orientation_xyzw.w, 1.0,)
    reference_wrist = derived.body.left_wrist_displacement_normalized_body
    moved_body = replace(
        derived.body,
        left_wrist_displacement_normalized_body = Vector3(
            x=reference_wrist.x + 0.20,
            y=reference_wrist.y - 0.10,
            z=reference_wrist.z + 0.10,
        ),
    )

    moved_human = replace(
        derived,
        frame_sequence_id = 43,
        measurement_timestamp_s = 30.100,
        body = moved_body,
    )

    mapped_result = retargeter.update(moved_human)

    assert mapped_result.valid
    assert mapped_result.active
    assert mapped_result.frame_sequence_id == 43
    assert_close (mapped_result.measurement_timestamp_s, 30.100)

    assert mapped_result.target_pose is not None
    assert not mapped_result.workspace_limited
    assert not mapped_result.speed_limited

    assert_close(mapped_result.target_pose.position.x, 0.70)
    assert_close(mapped_result.target_pose.position.y, 1.60)
    assert_close(mapped_result.target_pose.position.z, 3.40)

    assert_close(mapped_result.target_pose.orientation_xyzw.x, 0.0)
    assert_close(mapped_result.target_pose.orientation_xyzw.y, 0.0)
    assert_close(mapped_result.target_pose.orientation_xyzw.z, 0.0)
    assert_close(mapped_result.target_pose.orientation_xyzw.w, 1.0)

    print("H8 relative mapping and fixed orientation: PASS")

    deadband_retargeter = RelativeRetargeter(
        arm_side = ArmSide.LEFT,
        config = mapping_config,
    )
    deadband_retargeter.activate(derived, reference_robot_pose)

    deadband_body = replace(
        derived.body,
        left_wrist_displacement_normalized_body = Vector3(
            x = reference_wrist.x + 0.02,
            y = reference_wrist.y - 0.03,
            z = reference_wrist.z + 0.01,
        ),
    )

    deadband_human = replace(
        derived,
        frame_sequence_id = 43,
        measurement_timestamp_s = 30.100,
        body = deadband_body,
    )

    deadband_result = deadband_retargeter.update(deadband_human)
    assert deadband_result.valid
    assert deadband_result.target_pose is not None

    assert_close(deadband_result.target_pose.position.x, 1.0)
    assert_close(deadband_result.target_pose.position.y, 2.0)
    assert_close(deadband_result.target_pose.position.z, 3.0)

    print("H8 normalized deadband handling: PASS")

    workspace_config = RetargetingConfig(
        axis_mapping = AxisMapping.identity(),
        scale_robot_per_normalized_body = Vector3(x = 5.0, y = 5.0, z = 5.0),
        deadband_normalized_body = Vector3(x = 0.0, y = 0.0, z = 0.0),
        workspace_bounds=CartesianBounds(
            minimum=Vector3(x = 0.0, y = 0.0, z = 0.0),
            maximum=Vector3(x = 1.1, y = 2.1, z = 3.1),
        ),
        max_cartesian_speed_robot_per_s = None,
    )

    workspace_retargeter = RelativeRetargeter(
        arm_side = ArmSide.LEFT,
        config = workspace_config,
    )

    workspace_retargeter.activate(derived, reference_robot_pose)
    outside_body = replace(
        derived.body,
        left_wrist_displacement_normalized_body = Vector3(
            x = reference_wrist.x + 1.0,
            y = reference_wrist.y + 1.0,
            z = reference_wrist.z + 1.0,
        ),
    )

    outside_human = replace(
        derived,
        frame_sequence_id = 43,
        measurement_timestamp_s = 30.100,
        body = outside_body,
    )

    workspace_result = workspace_retargeter.update(outside_human)
    assert workspace_result.valid
    assert workspace_result.workspace_limited
    assert workspace_result.target_pose is not None

    assert_close(workspace_result.target_pose.position.x, 1.1)
    assert_close(workspace_result.target_pose.position.y, 2.1)
    assert_close(workspace_result.target_pose.position.z, 3.1)

    print("H8 workspace-limit handling: PASS")

    speed_config = RetargetingConfig(
        axis_mapping = AxisMapping.identity(),
        scale_robot_per_normalized_body = Vector3(x = 10.0, y = 10.0, z = 10.0),
        deadband_normalized_body = Vector3(x = 0.0, y = 0.0, z = 0.0),
        workspace_bounds = CartesianBounds(
            minimum = Vector3(x = -100.0, y = -100.0, z = -100.0),
            maximum = Vector3(x = 100.0, y = 100.0, z = 100.0),
        ),
        max_cartesian_speed_robot_per_s = 1.0,
    )

    speed_retargeter = RelativeRetargeter(arm_side = ArmSide.LEFT, config = speed_config)
    speed_retargeter.activate(derived, reference_robot_pose)

    fast_body = replace(
        derived.body,
        left_wrist_displacement_normalized_body = Vector3(
            x = reference_wrist.x + 1.0,
            y = reference_wrist.y,
            z = reference_wrist.z,
        ),
    )

    fast_human = replace(
        derived,
        frame_sequence_id = 43,
        measurement_timestamp_s = 30.100,
        body = fast_body,
    )

    speed_result = speed_retargeter.update(fast_human)
    assert speed_result.valid
    assert speed_result.speed_limited
    assert speed_result.target_pose is not None

    speed_step_x = speed_result.target_pose.position.x - reference_robot_pose.position.x

    assert_close(speed_step_x, 0.10, tolerance = 1e-9)

    print("H8 Cartesian speed-limit handling: PASS")

    invalid_human = replace(
        derived,
        frame_sequence_id = 43,
        measurement_timestamp_s = 30.100,
        body = None,
    )

    invalid_retargeter = RelativeRetargeter(
        arm_side = ArmSide.RIGHT,
        config = mapping_config,
    )

    inactive_result = invalid_retargeter.update(invalid_human)

    assert not inactive_result.valid
    assert not inactive_result.active
    assert inactive_result.target_pose is None
    assert "retargeting_inactive" in inactive_result.reasons

    invalid_retargeter.activate(right_derived, reference_robot_pose)
    missing_result = invalid_retargeter.update(invalid_human)

    assert not missing_result.valid
    assert missing_result.active
    assert missing_result.target_pose is None
    assert "body_derived_kinematics_unavailable" in missing_result.reasons

    print("H8 fail-closed invalid-input handling: PASS")

    invalid_order_retargeter = RelativeRetargeter(
        arm_side = ArmSide.RIGHT,
        config = mapping_config,
    )

    invalid_order_retargeter.activate(right_derived, reference_robot_pose)
    first_invalid_order_result = invalid_order_retargeter.update(invalid_human)

    assert not first_invalid_order_result.valid
    assert first_invalid_order_result.active

    with _expect_value_error('H8 accepted the same invalid frame twice.'):
        invalid_order_retargeter.update(invalid_human)

    print("H8 invalid-frame ordering guard: PASS")

    ordering_retargeter = RelativeRetargeter(
        arm_side = ArmSide.LEFT,
        config = mapping_config,
    )

    ordering_retargeter.activate(derived, reference_robot_pose)

    with _expect_value_error('H8 accepted a non-increasing timestamp.'):
        ordering_retargeter.update(
            replace(
                derived,
                frame_sequence_id = 42,
                measurement_timestamp_s = 30.000,
            )
        )

    print("H8 timestamp ordering guard: PASS")

    with _expect_value_error('Invalid axis mapping was accepted.'):
        AxisMapping(
            rows = (
                (1.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        )

    with _expect_value_error('Invalid retargeting configuration was accepted.'):
        RetargetingConfig(
            axis_mapping = AxisMapping.identity(),
            scale_robot_per_normalized_body = Vector3(x = 1.0, y = 1.0, z = 1.0),
            deadband_normalized_body=Vector3(x = -0.1, y = 0.0, z = 0.0),
            workspace_bounds=CartesianBounds(
                minimum=Vector3(x = -1.0, y = -1.0, z = -1.0),
                maximum=Vector3(x = 1.0, y = 1.0, z = 1.0),
            ),
        )

    print("H8 configuration guards: PASS")
    return mapped_result


def _run_p1_checks(mapped_result):
    supervisor_config = CommandSupervisorConfig(
        consecutive_valid_required = 3,
        max_human_age_s = 0.20,
        max_robot_state_age_s = 0.20,
        max_dynamic_transform_age_s = 0.20,
        max_target_age_s = 0.20,
        max_human_robot_skew_s = 0.05,
        max_human_transform_skew_s = 0.05,
        max_human_target_skew_s = 0.01,
    )

    supervisor = CommandSupervisor(supervisor_config)
    supervisor_cycle_id = 0

    def make_supervisor_cycle(
        decision_timestamp_s: float,
        *,
        human_valid: bool = True,
        human_frame_sequence_id: int = 100,
        human_timestamp_s: float | None = None,
        robot_state_valid: bool = True,
        robot_timestamp_s: float | None = None,
        transform_available: bool = True,
        transform_timestamp_s: float | None = None,
        retargeting_result = None,
        gripper_aperture_result = None,
        controller_ready: bool = True,
        activation_requested: bool = False,
        enable_requested: bool = False,
        disable_requested: bool = False,
        fault_reset_requested: bool = False,
        controller_fault: bool = False,
        persistent_inconsistency: bool = False,
        external_stop: bool = False,
    ):
        nonlocal supervisor_cycle_id

        if human_timestamp_s is None:
            human_timestamp_s = decision_timestamp_s - 0.01

        if robot_timestamp_s is None:
            robot_timestamp_s = human_timestamp_s

        if retargeting_result is not None:
            retargeting_result = replace(
                retargeting_result,
                frame_sequence_id = human_frame_sequence_id,
                measurement_timestamp_s = human_timestamp_s,
            )

        if gripper_aperture_result is not None:
            gripper_aperture_result = replace(
                gripper_aperture_result,
                frame_sequence_id = human_frame_sequence_id,
                measurement_timestamp_s = human_timestamp_s
            )

        cycle = SupervisorCycleInput(
            cycle_sequence_id = supervisor_cycle_id,
            decision_timestamp_s = decision_timestamp_s,
            human_valid=human_valid,
            human_frame_sequence_id = human_frame_sequence_id,
            human_measurement_timestamp_s = human_timestamp_s,
            robot_state_valid = robot_state_valid,
            robot_state_timestamp_s = robot_timestamp_s,
            transform_available = transform_available,
            transform_timestamp_s = transform_timestamp_s,
            retargeting_result = retargeting_result,
            controller_ready = controller_ready,
            gripper_aperture_result = gripper_aperture_result,
            activation_requested = activation_requested,
            enable_requested = enable_requested,
            disable_requested = disable_requested,
            fault_reset_requested = fault_reset_requested,
            controller_fault = controller_fault,
            persistent_inconsistency = persistent_inconsistency,
            external_stop = external_stop
        )

        supervisor_cycle_id += 1

        return cycle

    disabled_result = supervisor.update(make_supervisor_cycle(40.00))
    assert disabled_result.state == SupervisorState.DISABLED
    assert not disabled_result.motion_permitted
    assert disabled_result.permitted_target is None

    acquiring_start = supervisor.update(make_supervisor_cycle(40.10, activation_requested = True))
    assert acquiring_start.state == SupervisorState.ACQUIRING
    assert not acquiring_start.motion_permitted

    acquiring_1 = supervisor.update(make_supervisor_cycle(40.20))
    acquiring_2 = supervisor.update(make_supervisor_cycle(40.30))

    ready_result = supervisor.update(make_supervisor_cycle(40.40))
    assert acquiring_1.state == SupervisorState.ACQUIRING
    assert acquiring_2.state == SupervisorState.ACQUIRING
    assert ready_result.state == SupervisorState.READY
    assert ready_result.consecutive_valid_cycles == 3
    assert not ready_result.motion_permitted

    print("P1 DISABLED -> ACQUIRING -> READY gate: PASS")

    no_target_enable = supervisor.update(make_supervisor_cycle(40.50, enable_requested = True))

    assert no_target_enable.state == SupervisorState.READY
    assert not no_target_enable.motion_permitted
    assert "retargeting_missing" in no_target_enable.reasons
    active_result = supervisor.update(
        make_supervisor_cycle(
            40.60,
            retargeting_result = (mapped_result),
            enable_requested = True,
        )
    )

    assert active_result.state == SupervisorState.ACTIVE
    assert active_result.motion_permitted
    assert active_result.permitted_target is not None
    assert active_result.reasons == ()

    print("P1 READY -> ACTIVE explicit enable gate: PASS")

    hold_result = supervisor.update(
        make_supervisor_cycle(
            40.70,
            human_valid = False,
            retargeting_result = mapped_result
        )
    )

    assert hold_result.state == SupervisorState.HOLD
    assert not hold_result.motion_permitted
    assert hold_result.permitted_target is None
    assert "human_tracking_invalid" in hold_result.reasons

    recovery_1 = supervisor.update(
        make_supervisor_cycle(
            40.80,
            retargeting_result = mapped_result,
            enable_requested = True,
        )
    )

    recovery_2 = supervisor.update(
        make_supervisor_cycle(40.90,retargeting_result = mapped_result)
    )

    stable_hold = supervisor.update(
        make_supervisor_cycle(41.00, retargeting_result = mapped_result)
    )

    assert recovery_1.state == SupervisorState.HOLD
    assert not recovery_1.motion_permitted
    assert recovery_2.state == SupervisorState.HOLD
    assert stable_hold.state == SupervisorState.HOLD
    assert not stable_hold.motion_permitted
    assert "explicit_reenable_required" in stable_hold.reasons

    recovered_active = supervisor.update(
        make_supervisor_cycle(
            41.10,
            retargeting_result = mapped_result,
            enable_requested = True,
        )
    )

    assert recovered_active.state == SupervisorState.ACTIVE
    assert recovered_active.motion_permitted

    print("P1 HOLD recovery and explicit re-enable: PASS")

    stale_result = supervisor.update(
        make_supervisor_cycle(
            41.20,
            human_timestamp_s = 40.00,
            robot_timestamp_s = 41.19,
            retargeting_result = mapped_result
        )
    )

    assert stale_result.state == SupervisorState.HOLD
    assert not stale_result.motion_permitted
    assert "human_stale" in stale_result.reasons

    print("P1 stale-measurement fail-closed handling: PASS")

    supervisor.update(
        make_supervisor_cycle(
            41.30,
            retargeting_result = mapped_result,
        )
    )
    supervisor.update(
        make_supervisor_cycle(
            41.40,
            retargeting_result = mapped_result,
        )
    )
    supervisor.update(
        make_supervisor_cycle(
            41.50,
            retargeting_result = mapped_result,
        )
    )

    recovered_again = supervisor.update(
        make_supervisor_cycle(
            41.60,
            retargeting_result = mapped_result,
            enable_requested = True,
        )
    )

    assert recovered_again.state == SupervisorState.ACTIVE

    sync_failure = supervisor.update(
        make_supervisor_cycle(
            41.70,
            human_timestamp_s = 41.69,
            robot_timestamp_s = 41.55,
            retargeting_result = mapped_result,
        )
    )

    assert sync_failure.state == SupervisorState.HOLD
    assert not sync_failure.motion_permitted
    assert "human_robot_unsynchronized" in sync_failure.reasons

    print("P1 human/robot synchronization guard: PASS")

    disabled_from_hold = supervisor.update(
        make_supervisor_cycle(41.80, disable_requested = True)
    )

    assert disabled_from_hold.state == SupervisorState.DISABLED
    assert not disabled_from_hold.motion_permitted

    supervisor.update(
        make_supervisor_cycle(41.90, activation_requested = True)
    )
    supervisor.update(make_supervisor_cycle(42.00))
    supervisor.update(make_supervisor_cycle(42.10))
    supervisor.update(make_supervisor_cycle(42.20))

    active_before_fault = supervisor.update(
        make_supervisor_cycle(
            42.30,
            retargeting_result = mapped_result,
            enable_requested = True,
        )
    )

    assert active_before_fault.state == SupervisorState.ACTIVE

    fault_result = supervisor.update(
        make_supervisor_cycle(
            42.40,
            retargeting_result = mapped_result,
            external_stop = True,
        )
    )

    assert fault_result.state == SupervisorState.FAULT
    assert not fault_result.motion_permitted
    assert "external_stop" in fault_result.reasons

    fault_still_latched = supervisor.update(
        make_supervisor_cycle(42.50, fault_reset_requested = False)
    )

    assert fault_still_latched.state == SupervisorState.FAULT
    assert not fault_still_latched.motion_permitted
    assert "fault_reset_required" in fault_still_latched.reasons

    fault_reset = supervisor.update(
        make_supervisor_cycle(42.60, fault_reset_requested = True)
    )

    assert fault_reset.state == SupervisorState.DISABLED
    assert not fault_reset.motion_permitted

    print("P1 FAULT latch and manual reset: PASS")

    ordering_supervisor = CommandSupervisor(supervisor_config)

    ordered_cycle = SupervisorCycleInput(
        cycle_sequence_id = 0,
        decision_timestamp_s = 50.0,
        human_valid = True,
        human_frame_sequence_id = 1,
        human_measurement_timestamp_s = 49.99,
        robot_state_valid = True,
        robot_state_timestamp_s = 49.99,
        transform_available = True,
        transform_timestamp_s = None,
        retargeting_result = None,
        controller_ready = True,
    )

    ordering_supervisor.update(ordered_cycle)

    with _expect_value_error('P1 accepted a duplicate supervisor cycle.'):
        ordering_supervisor.update(ordered_cycle)

    print("P1 cycle ordering guard: PASS")

    with _expect_value_error('P1 accepted an invalid supervisor configuration.'):
        CommandSupervisorConfig(
            consecutive_valid_required = 0,
            max_human_age_s = 0.1,
            max_robot_state_age_s = 0.1,
            max_dynamic_transform_age_s = 0.1,
            max_target_age_s = 0.1,
            max_human_robot_skew_s = 0.1,
            max_human_transform_skew_s = 0.1,
            max_human_target_skew_s = 0.1,
        )

    print("P1 supervisor configuration guards: PASS")
    return supervisor_config, active_result

def _run_p2_checks(derived):
    gripper_config = GripperIntentConfig(
        close_threshold = 0.40,
        open_threshold = 0.80,
        consecutive_confirmations = 3,
    )

    gripper_tracker = GripperIntentTracker(
        hand_side = HandSide.RIGHT,
        config = gripper_config,
    )

    if derived.right_hand is None:
        raise AssertionError("P2 requires valid synthetic right-hand H6 geometry.")

    base_hand = derived.right_hand

    def hand_frame(
        frame_sequence_id: int,
        timestamp_s: float,
        pinch_ratio: float | None,
    ):
        if pinch_ratio is None:
            return replace(
                derived,
                frame_sequence_id = frame_sequence_id,
                measurement_timestamp_s = timestamp_s,
                right_hand = None,
            )

        return replace(
            derived,
            frame_sequence_id = frame_sequence_id,
            measurement_timestamp_s = timestamp_s,
            right_hand = replace(base_hand, pinch_ratio = pinch_ratio),
        )

    close_1 = gripper_tracker.update(hand_frame(100, 60.00, 0.30))
    close_2 = gripper_tracker.update(hand_frame(101, 60.10, 0.32))
    close_3 = gripper_tracker.update(hand_frame(102, 60.20, 0.35))

    assert close_1.stable_state == GripperIntentState.UNKNOWN
    assert not close_1.valid_for_control
    assert close_1.pending_state == GripperIntentState.CLOSED
    assert close_1.consecutive_pending == 1
    assert close_2.consecutive_pending == 2
    assert close_3.stable_state == GripperIntentState.CLOSED
    assert close_3.state_changed
    assert close_3.valid_for_control
    assert close_3.desired_state == GripperIntentState.CLOSED

    print("P2 initial CLOSED confirmation gate: PASS")

    hysteresis_result = gripper_tracker.update(
        hand_frame(103, 60.30, 0.60)
    )

    assert hysteresis_result.stable_state == GripperIntentState.CLOSED
    assert not hysteresis_result.state_changed
    assert hysteresis_result.valid_for_control
    assert hysteresis_result.desired_state == GripperIntentState.CLOSED
    assert hysteresis_result.pending_state is None

    print("P2 hysteresis deadband preservation: PASS")

    open_1 = gripper_tracker.update(hand_frame(104, 60.40, 0.90))
    interrupted = gripper_tracker.update(hand_frame(105, 60.50, None))

    assert open_1.pending_state == GripperIntentState.OPEN
    assert open_1.consecutive_pending == 1
    assert not interrupted.measurement_valid
    assert not interrupted.valid_for_control
    assert interrupted.desired_state is None
    assert interrupted.stable_state == GripperIntentState.CLOSED
    assert interrupted.pending_state is None
    assert interrupted.consecutive_pending == 0

    print("P2 invalid-hand command suppression/reset: PASS")

    open_2a = gripper_tracker.update(hand_frame(106, 60.60, 0.90))
    open_2b = gripper_tracker.update(hand_frame(107, 60.70, 0.95))
    open_2c = gripper_tracker.update(hand_frame(108, 60.80, 1.00))

    assert open_2a.consecutive_pending == 1
    assert open_2b.consecutive_pending == 2
    assert open_2c.stable_state == GripperIntentState.OPEN
    assert open_2c.state_changed
    assert open_2c.valid_for_control
    assert open_2c.desired_state == GripperIntentState.OPEN

    print("P2 OPEN confirmation after reacquisition: PASS")

    close_candidate = gripper_tracker.update(
        hand_frame(109, 60.90, 0.30)
    )
    return_to_open = gripper_tracker.update(
        hand_frame(110, 61.00, 0.90)
    )

    assert close_candidate.pending_state == GripperIntentState.CLOSED
    assert return_to_open.stable_state == GripperIntentState.OPEN
    assert return_to_open.pending_state is None
    assert return_to_open.consecutive_pending == 0
    assert return_to_open.desired_state == GripperIntentState.OPEN

    print("P2 interrupted transition cancellation: PASS")

    ordering_gripper = GripperIntentTracker(
        hand_side = HandSide.RIGHT,
        config = gripper_config,
    )
    ordered_hand_frame = hand_frame(200, 70.0, 0.9)
    ordering_gripper.update(ordered_hand_frame)

    with _expect_value_error('P2 accepted a duplicate hand-intent frame.'):
        ordering_gripper.update(ordered_hand_frame)

    print("P2 gripper-intent ordering guard: PASS")

    with _expect_value_error('P2 accepted reversed hysteresis thresholds.'):
        GripperIntentConfig(
            close_threshold = 0.8,
            open_threshold = 0.4,
            consecutive_confirmations = 3,
        )

    with _expect_value_error('P2 accepted zero confirmation frames.'):
        GripperIntentConfig(
            close_threshold = 0.4,
            open_threshold = 0.8,
            consecutive_confirmations = 0,
        )

    print("P2 gripper-intent configuration guards: PASS")
    return hand_frame


def _run_p4b_checks(derived, hand_frame):
    aperture_config = GripperApertureConfig(
        closed_reference_ratio = 0.40,
        open_reference_ratio = 1.40,
    )
    aperture_tracker = GripperApertureTracker(
        hand_side = HandSide.RIGHT,
        config = aperture_config,
    )

    closed_aperture = aperture_tracker.update(
        hand_frame(300, 80.00, 0.40)
    )
    midpoint_aperture = aperture_tracker.update(
        hand_frame(301, 80.10, 0.90)
    )
    open_aperture = aperture_tracker.update(
        hand_frame(302, 80.20, 1.40)
    )

    assert closed_aperture.measurement_valid
    assert closed_aperture.normalized_aperture == 0.0
    assert closed_aperture.fully_closed
    assert not closed_aperture.fully_open

    assert midpoint_aperture.measurement_valid
    assert np.isclose(midpoint_aperture.normalized_aperture, 0.5)
    assert not midpoint_aperture.fully_closed
    assert not midpoint_aperture.fully_open

    assert open_aperture.measurement_valid
    assert open_aperture.normalized_aperture == 1.0
    assert open_aperture.fully_open
    assert not open_aperture.fully_closed

    print("P4B continuous aperture normalization: PASS")

    below_closed = aperture_tracker.update(
        hand_frame(303, 80.30, 0.10)
    )
    above_open = aperture_tracker.update(
        hand_frame(304, 80.40, 1.80)
    )

    assert below_closed.normalized_aperture == 0.0
    assert below_closed.fully_closed
    assert above_open.normalized_aperture == 1.0
    assert above_open.fully_open

    print("P4B aperture clamping/endpoints: PASS")

    invalid_hand_aperture = aperture_tracker.update(
        hand_frame(305, 80.50, None)
    )
    assert not invalid_hand_aperture.measurement_valid
    assert invalid_hand_aperture.normalized_aperture is None
    assert not invalid_hand_aperture.fully_open
    assert not invalid_hand_aperture.fully_closed
    assert "right_hand_geometry_unavailable" in invalid_hand_aperture.reasons

    invalid_ratio_aperture = aperture_tracker.update(
        hand_frame(306, 80.60, float("nan"))
    )
    assert not invalid_ratio_aperture.measurement_valid
    assert invalid_ratio_aperture.normalized_aperture is None
    assert "pinch_ratio_invalid" in invalid_ratio_aperture.reasons

    negative_ratio_aperture = aperture_tracker.update(
        hand_frame(307, 80.70, -0.01)
    )
    assert not negative_ratio_aperture.measurement_valid
    assert negative_ratio_aperture.normalized_aperture is None
    assert negative_ratio_aperture.pinch_ratio == -0.01
    assert "pinch_ratio_negative" in negative_ratio_aperture.reasons

    print("P4B invalid-measurement suppression: PASS")

    if derived.left_hand is None:
        raise AssertionError("P4B requires valid synthetic left-hand H6 geometry.")

    left_base_hand = derived.left_hand
    left_side_tracker = GripperApertureTracker(
        hand_side = HandSide.LEFT,
        config = aperture_config,
    )
    left_only_frame = replace(
        derived,
        frame_sequence_id = 400,
        measurement_timestamp_s = 90.0,
        left_hand = replace(left_base_hand, pinch_ratio = 0.90),
        right_hand = None,
    )
    left_side_result = left_side_tracker.update(left_only_frame)
    assert left_side_result.measurement_valid
    assert left_side_result.hand_side == HandSide.LEFT
    assert np.isclose(left_side_result.normalized_aperture, 0.5)

    print("P4B selected-hand isolation: PASS")

    ordering_aperture = GripperApertureTracker(
        hand_side = HandSide.RIGHT,
        config = aperture_config,
    )
    ordered_aperture_frame = hand_frame(500, 100.0, 0.9)
    ordering_aperture.update(ordered_aperture_frame)

    with _expect_value_error('P4B accepted a duplicate aperture frame.'):
        ordering_aperture.update(ordered_aperture_frame)

    same_timestamp_frame = hand_frame(501, 100.0, 0.9)
    with _expect_value_error('P4B accepted a non-increasing aperture timestamp.'):
        ordering_aperture.update(same_timestamp_frame)

    invalid_ordering_aperture = GripperApertureTracker(
        hand_side = HandSide.RIGHT,
        config = aperture_config,
    )
    ordered_invalid_frame = hand_frame(600, 110.0, None)
    invalid_ordering_aperture.update(ordered_invalid_frame)
    with _expect_value_error('P4B did not consume ordering state for an invalid hand frame.'):
        invalid_ordering_aperture.update(ordered_invalid_frame)

    print("P4B gripper-aperture ordering guard: PASS")

    invalid_aperture_configs = (
        (-0.1, 1.0),
        (0.4, 0.4),
        (0.8, 0.4),
        (0.4, float("inf")),
    )
    for closed_reference, open_reference in invalid_aperture_configs:
        with _expect_value_error('P4B accepted an invalid aperture calibration configuration.'):
            GripperApertureConfig(
                closed_reference_ratio = closed_reference,
                open_reference_ratio = open_reference,
            )

    print("P4B gripper-aperture configuration guards: PASS")
    return aperture_config

def _run_p4d_checks(
    mapped_result,
    supervisor_config,
    active_result,
    aperture_config,
    hand_frame,
):
    def p4d_supervisor_config(policy: GripperLossPolicy) -> CommandSupervisorConfig:
        return CommandSupervisorConfig(
            consecutive_valid_required = 3,
            max_human_age_s = 0.20,
            max_robot_state_age_s = 0.20,
            max_dynamic_transform_age_s = 0.20,
            max_target_age_s = 0.20,
            max_human_robot_skew_s = 0.05,
            max_human_transform_skew_s = 0.05,
            max_human_target_skew_s = 0.01,
            gripper_loss_policy = policy,
        )

    p4d_aperture_tracker = GripperApertureTracker(
        hand_side = HandSide.RIGHT,
        config = aperture_config,
    )
    p4d_valid_aperture = p4d_aperture_tracker.update(
        hand_frame(700, 120.00, 0.90)
    )
    p4d_invalid_aperture = p4d_aperture_tracker.update(
        hand_frame(701, 120.10, None)
    )

    assert p4d_valid_aperture.measurement_valid
    assert np.isclose(p4d_valid_aperture.normalized_aperture, 0.5)
    assert not p4d_invalid_aperture.measurement_valid

    def p4d_cycle(
        cycle_id: int,
        decision_timestamp_s: float,
        *,
        gripper_aperture_result,
        activation_requested: bool = False,
        enable_requested: bool = False,
    ) -> SupervisorCycleInput:
        human_frame_sequence_id = 900 + cycle_id
        human_timestamp_s = decision_timestamp_s - 0.01

        target = replace(
            mapped_result,
            frame_sequence_id = human_frame_sequence_id,
            measurement_timestamp_s = human_timestamp_s,
        )

        gripper = (
            None if gripper_aperture_result is None
            else replace(
                gripper_aperture_result,
                frame_sequence_id = human_frame_sequence_id,
                measurement_timestamp_s = human_timestamp_s,
            )
        )

        return SupervisorCycleInput(
            cycle_sequence_id = cycle_id,
            decision_timestamp_s = decision_timestamp_s,
            human_valid = True,
            human_frame_sequence_id = human_frame_sequence_id,
            human_measurement_timestamp_s = human_timestamp_s,
            robot_state_valid = True,
            robot_state_timestamp_s = human_timestamp_s,
            transform_available = True,
            transform_timestamp_s = None,
            retargeting_result = target,
            controller_ready = True,
            gripper_aperture_result = gripper,
            activation_requested = activation_requested,
            enable_requested = enable_requested,
        )

    hold_policy_supervisor = CommandSupervisor(
        p4d_supervisor_config(GripperLossPolicy.HOLD_TELEOP)
    )

    p4d_acquiring = hold_policy_supervisor.update(
        p4d_cycle(
            0, 120.20,
            gripper_aperture_result = p4d_valid_aperture,
            activation_requested = True,
        )
    )
    p4d_acquiring_1 = hold_policy_supervisor.update(
        p4d_cycle(
            1, 120.30,
            gripper_aperture_result = p4d_valid_aperture,
        )
    )
    p4d_acquiring_2 = hold_policy_supervisor.update(
        p4d_cycle(
            2, 120.40,
            gripper_aperture_result = p4d_valid_aperture,
        )
    )
    p4d_ready = hold_policy_supervisor.update(
        p4d_cycle(
            3, 120.50,
            gripper_aperture_result = p4d_valid_aperture,
        )
    )

    for non_active_result in (
        p4d_acquiring,
        p4d_acquiring_1,
        p4d_acquiring_2,
        p4d_ready,
    ):
        assert not non_active_result.motion_permitted
        assert non_active_result.permitted_target is None
        assert not non_active_result.gripper_command_permitted
        assert non_active_result.permitted_gripper_aperture is None

    assert p4d_ready.state == SupervisorState.READY

    p4d_active = hold_policy_supervisor.update(
        p4d_cycle(
            4, 120.60,
            gripper_aperture_result = p4d_valid_aperture,
            enable_requested = True,
        )
    )

    assert p4d_active.state == SupervisorState.ACTIVE
    assert p4d_active.motion_permitted
    assert p4d_active.permitted_target is not None
    assert p4d_active.gripper_command_permitted
    assert np.isclose( p4d_active.permitted_gripper_aperture, 0.5)
    assert not p4d_active.reasons
    assert not p4d_active.gripper_reasons

    print("P4D synchronized arm+aperture ACTIVE permission: PASS")
    print("P4D no command outside ACTIVE: PASS")

    p4d_hand_loss = hold_policy_supervisor.update(
        p4d_cycle(
            5, 120.70,
            gripper_aperture_result = p4d_invalid_aperture,
        )
    )

    assert p4d_hand_loss.state == SupervisorState.HOLD
    assert not p4d_hand_loss.motion_permitted
    assert p4d_hand_loss.permitted_target is None
    assert not p4d_hand_loss.gripper_command_permitted
    assert p4d_hand_loss.permitted_gripper_aperture is None
    assert "gripper_aperture_invalid" in p4d_hand_loss.reasons
    assert "gripper_aperture_invalid" in p4d_hand_loss.gripper_reasons

    print("P4D HOLD policy gripper-loss fail-closed handling: PASS")

    p4d_recovery_1 = hold_policy_supervisor.update(
        p4d_cycle(
            6, 120.80,
            gripper_aperture_result = p4d_valid_aperture,
        )
    )
    p4d_recovery_2 = hold_policy_supervisor.update(
        p4d_cycle(
            7, 120.90,
            gripper_aperture_result = p4d_valid_aperture,
        )
    )
    p4d_recovered_waiting = hold_policy_supervisor.update(
        p4d_cycle(
            8, 121.00,
            gripper_aperture_result = p4d_valid_aperture,
        )
    )

    assert p4d_recovery_1.state == SupervisorState.HOLD
    assert p4d_recovery_2.state == SupervisorState.HOLD
    assert p4d_recovered_waiting.state == SupervisorState.HOLD
    assert "explicit_reenable_required" in p4d_recovered_waiting.reasons
    assert not p4d_recovered_waiting.motion_permitted
    assert not p4d_recovered_waiting.gripper_command_permitted

    p4d_reenabled = hold_policy_supervisor.update(
        p4d_cycle(
            9, 121.10,
            gripper_aperture_result = p4d_valid_aperture,
            enable_requested = True,
        )
    )

    assert p4d_reenabled.state == SupervisorState.ACTIVE
    assert p4d_reenabled.motion_permitted
    assert p4d_reenabled.gripper_command_permitted

    print("P4D gripper-loss recovery and explicit re-enable: PASS")

    mismatched_cycle = p4d_cycle(
        10, 121.20,
        gripper_aperture_result = p4d_valid_aperture,
    )
    assert mismatched_cycle.gripper_aperture_result is not None
    mismatched_cycle = replace(
        mismatched_cycle,
        gripper_aperture_result = replace(
            mismatched_cycle.gripper_aperture_result,
            frame_sequence_id = mismatched_cycle.human_frame_sequence_id + 1
        ),
    )
    p4d_mismatch = hold_policy_supervisor.update(mismatched_cycle)

    assert p4d_mismatch.state == SupervisorState.HOLD
    assert not p4d_mismatch.motion_permitted
    assert not p4d_mismatch.gripper_command_permitted
    assert "human_gripper_frame_mismatch" in p4d_mismatch.reasons

    print("P4D human/gripper synchronization guard: PASS")

    arm_only_supervisor = CommandSupervisor(
        p4d_supervisor_config(GripperLossPolicy.ALLOW_ARM_ONLY)
    )

    arm_only_supervisor.update(
        p4d_cycle(
            0,
            130.00,
            gripper_aperture_result = None,
            activation_requested = True,
        )
    )
    arm_only_supervisor.update(
        p4d_cycle(
            1, 130.10,
            gripper_aperture_result = None,
        )
    )
    arm_only_supervisor.update(
        p4d_cycle(
            2, 130.20,
            gripper_aperture_result = None,
        )
    )
    arm_only_ready = arm_only_supervisor.update(
        p4d_cycle(
            3, 130.30,
            gripper_aperture_result = None,
        )
    )
    assert arm_only_ready.state == SupervisorState.READY

    arm_only_active = arm_only_supervisor.update(
        p4d_cycle(
            4, 130.40,
            gripper_aperture_result = p4d_valid_aperture,
            enable_requested = True,
        )
    )
    assert arm_only_active.state == SupervisorState.ACTIVE
    assert arm_only_active.motion_permitted
    assert arm_only_active.gripper_command_permitted

    arm_only_hand_loss = arm_only_supervisor.update(
        p4d_cycle(
            5, 130.50,
            gripper_aperture_result = p4d_invalid_aperture,
        )
    )

    assert arm_only_hand_loss.state == SupervisorState.ACTIVE
    assert arm_only_hand_loss.motion_permitted
    assert arm_only_hand_loss.permitted_target is not None
    assert not arm_only_hand_loss.gripper_command_permitted
    assert arm_only_hand_loss.permitted_gripper_aperture is None
    assert not arm_only_hand_loss.reasons
    assert "gripper_aperture_invalid" in arm_only_hand_loss.gripper_reasons

    arm_only_missing = arm_only_supervisor.update(
        p4d_cycle(
            6, 130.60,
            gripper_aperture_result = None,
        )
    )
    assert arm_only_missing.state == SupervisorState.ACTIVE
    assert arm_only_missing.motion_permitted
    assert not arm_only_missing.gripper_command_permitted
    assert "gripper_aperture_missing" in arm_only_missing.gripper_reasons

    out_of_range_cycle = p4d_cycle(
        7, 130.70,
        gripper_aperture_result = p4d_valid_aperture,
    )
    assert out_of_range_cycle.gripper_aperture_result is not None
    out_of_range_cycle = replace(
        out_of_range_cycle,
        gripper_aperture_result = replace(
            out_of_range_cycle.gripper_aperture_result,
            normalized_aperture = 1.01,
        ),
    )
    arm_only_out_of_range = arm_only_supervisor.update(out_of_range_cycle)
    assert arm_only_out_of_range.state == SupervisorState.ACTIVE
    assert arm_only_out_of_range.motion_permitted
    assert not arm_only_out_of_range.gripper_command_permitted
    assert "gripper_aperture_value_out_of_range" in arm_only_out_of_range.gripper_reasons

    print("P4D ALLOW_ARM_ONLY gripper suppression policy: PASS")
    print("P4D gripper-aperture defensive range guard: PASS")

    assert supervisor_config.gripper_loss_policy is None
    assert not active_result.gripper_command_permitted
    assert active_result.permitted_gripper_aperture is None

    print("P4D legacy P1 supervisor compatibility: PASS")

    with _expect_value_error('P4D accepted a non-enum gripper-loss policy.'):
        CommandSupervisorConfig(
            consecutive_valid_required = 3,
            max_human_age_s = 0.20,
            max_robot_state_age_s = 0.20,
            max_dynamic_transform_age_s = 0.20,
            max_target_age_s = 0.20,
            max_human_robot_skew_s = 0.05,
            max_human_transform_skew_s = 0.05,
            max_human_target_skew_s = 0.01,
            gripper_loss_policy = "hold_teleop",
        )

    print("P4D gripper-loss policy configuration guard: PASS")


def _print_results() -> None:
    print("-" * 60)
    print("H3 RESULT: PASS")
    print("H4 RESULT: PASS")
    print("H4 TEMPORAL RESULT: PASS")
    print("HT1 POSE-GUIDED HAND ASSOCIATION RESULT: PASS")
    print("H5 RESULT: PASS")
    print("H6 RESULT: PASS")
    print("HT3 HAND-GEOMETRY PLAUSIBILITY RESULT: PASS")
    print("HT4 PALM-LENGTH PINCH NORMALIZATION RESULT: PASS")
    print("H8 RESULT: PASS")
    print("P1 SUPERVISOR RESULT: PASS")
    print("P2 GRIPPER INTENT RESULT: PASS")
    print("P4A SELECTED-ARM WAIST-UP RESULT: PASS")
    print("P4B CONTINUOUS APERTURE RESULT: PASS")
    print("P4D INTEGRATED SUPERVISOR RESULT: PASS")
    print(
        "Robot-independent retargeting, supervision, binary pinch "
        "intent, continuous normalized aperture and integrated arm/gripper "
        "permission passed synthetic validity, dropout and ordering checks.")

def main():
    print("=" * 60)
    print("H3/H4/H5/H6/H8/P1/P2/P4A/P4B/P4D - Human Tracking Synthetic Check")
    print("=" * 60)

    fixtures = _run_h3_h4_checks()
    _run_ht1_checks(fixtures)
    _run_h5_checks(fixtures)

    derived, right_derived = _run_h6_checks(fixtures.observation)
    mapped_result = _run_h8_checks(derived, right_derived)
    supervisor_config, active_result = _run_p1_checks(mapped_result)
    hand_frame = _run_p2_checks(derived)
    aperture_config = _run_p4b_checks(derived, hand_frame)
    _run_p4d_checks(
        mapped_result,
        supervisor_config,
        active_result,
        aperture_config,
        hand_frame,
    )
    _print_results()

if __name__ == "__main__":
    main()