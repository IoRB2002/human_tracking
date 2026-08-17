from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from human_tracking.acquisition import (
    CameraFrame,
)
from human_tracking.mediapipe_adapter import (
    MediaPipeObservationAdapter,
)
from human_tracking.mediapipe_backend import (
    MediaPipeTrackingOutput,
)
from human_tracking.observation import (
    Landmark,
    LandmarkSet,
    ObservationValidationConfig,
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


def fake_landmarks(
    count: int,
    with_confidence: bool,
    visibility: float = 0.90,
    presence: float = 0.95,
):
    landmarks = []

    for index in range(count):
        landmark = SimpleNamespace(
            x=0.01 * index,
            y=0.02 * index,
            z=-0.001 * index,
        )

        if with_confidence:
            landmark.visibility = visibility
            landmark.presence = presence

        landmarks.append(landmark)

    return landmarks


def fake_handedness(
    name: str,
    score: float,
):
    return [
        SimpleNamespace(
            category_name=name,
            score=score,
        )
    ]


def make_frame():
    return CameraFrame(
        sequence_id=7,
        measurement_timestamp_s=123.456,
        image_bgr=np.zeros(
            (480, 640, 3),
            dtype=np.uint8,
        ),
        image_width_px=640,
        image_height_px=480,
    )


def make_complete_output():
    return MediaPipeTrackingOutput(
        frame_sequence_id=7,
        measurement_timestamp_s=123.456,
        mediapipe_timestamp_ms=123456,
        pose_result=SimpleNamespace(
            pose_landmarks=[
                fake_landmarks(
                    33,
                    with_confidence=True,
                )
            ],
            pose_world_landmarks=[
                fake_landmarks(
                    33,
                    with_confidence=True,
                )
            ],
        ),
        hand_result=SimpleNamespace(
            hand_landmarks=[
                fake_landmarks(
                    21,
                    with_confidence=False,
                ),
                fake_landmarks(
                    21,
                    with_confidence=False,
                ),
            ],
            hand_world_landmarks=[
                fake_landmarks(
                    21,
                    with_confidence=False,
                ),
                fake_landmarks(
                    21,
                    with_confidence=False,
                ),
            ],
            handedness=[
                fake_handedness(
                    "Left",
                    0.97,
                ),
                fake_handedness(
                    "Right",
                    0.96,
                ),
            ],
        ),
        processing_duration_s=0.05,
    )


def shifted_observation(
    source,
    sequence_id: int,
    timestamp_s: float,
):
    return replace(
        source,
        frame_sequence_id=sequence_id,
        measurement_timestamp_s=timestamp_s,
    )


def with_body_landmark_x(
    source,
    landmark_index: int,
    x: float,
    sequence_id: int,
    timestamp_s: float,
):
    body_set = source.body_image_landmarks

    if body_set is None:
        raise ValueError(
            "Source observation has no body landmarks."
        )

    landmarks = list(
        body_set.landmarks
    )

    landmarks[landmark_index] = replace(
        landmarks[landmark_index],
        x=x,
    )

    changed_body_set = LandmarkSet(
        coordinate_space=(
            body_set.coordinate_space
        ),
        landmarks=tuple(landmarks),
    )

    return replace(
        source,
        frame_sequence_id=sequence_id,
        measurement_timestamp_s=timestamp_s,
        body_image_landmarks=changed_body_set,
    )



def landmark_set_with_positions(
    source: LandmarkSet,
    positions: dict[int, tuple[float, float, float]],
) -> LandmarkSet:
    landmarks = list(
        source.landmarks
    )

    for index, position in positions.items():
        current = landmarks[index]

        landmarks[index] = Landmark(
            index=current.index,
            name=current.name,
            x=position[0],
            y=position[1],
            z=position[2],
            visibility=current.visibility,
            presence=current.presence,
        )

    return LandmarkSet(
        coordinate_space=(
            source.coordinate_space
        ),
        landmarks=tuple(landmarks),
    )


def assert_close(
    actual: float,
    expected: float,
    tolerance: float = 1e-9,
):
    if abs(actual - expected) > tolerance:
        raise AssertionError(
            f"Expected {expected}, received {actual}."
        )

def main():
    print("=" * 60)
    print("H3/H4/H5/H6/H8 - Human Tracking Synthetic Check")
    print("=" * 60)

    frame = make_frame()
    adapter = MediaPipeObservationAdapter()

    observation = adapter.convert(
        frame,
        make_complete_output(),
    )

    assert observation.frame_sequence_id == 7
    assert observation.measurement_timestamp_s == 123.456
    assert observation.image_width_px == 640
    assert observation.image_height_px == 480

    assert observation.body_image_landmarks is not None
    assert len(observation.body_image_landmarks) == 33

    assert (
        observation
        .body_image_landmarks
        .landmarks[11]
        .name
        == "left_shoulder"
    )

    assert len(observation.left_hands) == 1
    assert len(observation.right_hands) == 1
    assert len(observation.unknown_hands) == 0

    assert (
        len(
            observation
            .left_hands[0]
            .image_landmarks
        )
        == 21
    )

    assert (
        observation
        .left_hands[0]
        .image_landmarks
        .landmarks[0]
        .name
        == "wrist"
    )

    serialized = observation.to_json()

    assert (
        '"measurement_timestamp_s":123.456'
        in serialized
    )

    assert (
        '"coordinate_space":"normalized_image"'
        in serialized
    )

    print("H3 complete observation: PASS")
    print("H3 timestamp preservation: PASS")
    print("H3 body schema: PASS")
    print("H3 hand schema: PASS")
    print("H3 left/right separation: PASS")
    print("H3 JSON serialization: PASS")

    missing_output = MediaPipeTrackingOutput(
        frame_sequence_id=7,
        measurement_timestamp_s=123.456,
        mediapipe_timestamp_ms=123456,
        pose_result=SimpleNamespace(
            pose_landmarks=[],
            pose_world_landmarks=[],
        ),
        hand_result=SimpleNamespace(
            hand_landmarks=[],
            hand_world_landmarks=[],
            handedness=[],
        ),
        processing_duration_s=0.05,
    )

    missing_observation = adapter.convert(
        frame,
        missing_output,
    )

    assert (
        missing_observation
        .body_image_landmarks
        is None
    )
    assert (
        missing_observation
        .body_world_landmarks
        is None
    )
    assert not missing_observation.left_hands
    assert not missing_observation.right_hands
    assert not missing_observation.unknown_hands

    print("H3 explicit missing data: PASS")

    config = ObservationValidationConfig(
        min_body_visibility=0.50,
        min_body_presence=0.50,
        min_handedness_score=0.50,
    )

    valid = evaluate_observation(
        observation,
        config,
    )

    assert valid.available_for_visualization
    assert valid.body_valid_for_control
    assert valid.left_hand_valid_for_control
    assert valid.right_hand_valid_for_control
    assert not valid.body_reasons
    assert not valid.left_hand_reasons
    assert not valid.right_hand_reasons

    print("H4 complete valid frame: PASS")

    missing_validity = evaluate_observation(
        missing_observation,
        config,
    )

    assert not (
        missing_validity
        .available_for_visualization
    )
    assert not (
        missing_validity
        .body_valid_for_control
    )
    assert not (
        missing_validity
        .left_hand_valid_for_control
    )
    assert not (
        missing_validity
        .right_hand_valid_for_control
    )

    assert (
        "body_missing"
        in missing_validity.body_reasons
    )
    assert (
        "left_hand_missing"
        in missing_validity.left_hand_reasons
    )
    assert (
        "right_hand_missing"
        in missing_validity.right_hand_reasons
    )

    print("H4 missing-data rejection: PASS")

    low_confidence_output = MediaPipeTrackingOutput(
        frame_sequence_id=7,
        measurement_timestamp_s=123.456,
        mediapipe_timestamp_ms=123456,
        pose_result=SimpleNamespace(
            pose_landmarks=[
                fake_landmarks(
                    33,
                    with_confidence=True,
                    visibility=0.20,
                    presence=0.20,
                )
            ],
            pose_world_landmarks=[
                fake_landmarks(
                    33,
                    with_confidence=True,
                    visibility=0.20,
                    presence=0.20,
                )
            ],
        ),
        hand_result=(
            make_complete_output()
            .hand_result
        ),
        processing_duration_s=0.05,
    )

    low_confidence_observation = (
        adapter.convert(
            frame,
            low_confidence_output,
        )
    )

    low_confidence_validity = (
        evaluate_observation(
            low_confidence_observation,
            config,
        )
    )

    assert (
        low_confidence_validity
        .available_for_visualization
    )
    assert not (
        low_confidence_validity
        .body_valid_for_control
    )
    assert (
        low_confidence_validity
        .left_hand_valid_for_control
    )
    assert (
        low_confidence_validity
        .right_hand_valid_for_control
    )

    assert any(
        reason.endswith(
            "_visibility_low"
        )
        for reason
        in low_confidence_validity.body_reasons
    )

    assert any(
        reason.endswith(
            "_presence_low"
        )
        for reason
        in low_confidence_validity.body_reasons
    )

    print(
        "H4 visualization/control separation: PASS"
    )

    low_hand_output = MediaPipeTrackingOutput(
        frame_sequence_id=7,
        measurement_timestamp_s=123.456,
        mediapipe_timestamp_ms=123456,
        pose_result=(
            make_complete_output()
            .pose_result
        ),
        hand_result=SimpleNamespace(
            hand_landmarks=[
                fake_landmarks(
                    21,
                    with_confidence=False,
                ),
                fake_landmarks(
                    21,
                    with_confidence=False,
                ),
            ],
            hand_world_landmarks=[
                fake_landmarks(
                    21,
                    with_confidence=False,
                ),
                fake_landmarks(
                    21,
                    with_confidence=False,
                ),
            ],
            handedness=[
                fake_handedness(
                    "Left",
                    0.25,
                ),
                fake_handedness(
                    "Right",
                    0.96,
                ),
            ],
        ),
        processing_duration_s=0.05,
    )

    low_hand_observation = adapter.convert(
        frame,
        low_hand_output,
    )

    low_hand_validity = evaluate_observation(
        low_hand_observation,
        config,
    )

    assert not (
        low_hand_validity
        .left_hand_valid_for_control
    )
    assert (
        low_hand_validity
        .right_hand_valid_for_control
    )
    assert (
        "left_handedness_score_low"
        in low_hand_validity.left_hand_reasons
    )

    print(
        "H4 handedness-confidence rejection: PASS"
    )

    try:
        ObservationValidationConfig(
            min_body_visibility=1.5,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Invalid threshold was accepted."
        )

    print("H4 threshold configuration check: PASS")

    temporal_config = TemporalTrackingConfig(
        consecutive_valid_required=3,
        dropout_timeout_s=0.25,
    )

    tracker = HumanTemporalTracker(
        temporal_config
    )

    temporal_1 = tracker.update(
        shifted_observation(
            observation,
            sequence_id=1,
            timestamp_s=10.000,
        ),
        valid,
    )

    assert (
        temporal_1.body.state
        == TrackingState.ACQUIRING
    )
    assert not (
        temporal_1.body.valid_for_control
    )

    temporal_2 = tracker.update(
        shifted_observation(
            observation,
            sequence_id=2,
            timestamp_s=10.067,
        ),
        valid,
    )

    assert (
        temporal_2.body.state
        == TrackingState.ACQUIRING
    )
    assert not (
        temporal_2.body.valid_for_control
    )

    temporal_3 = tracker.update(
        shifted_observation(
            observation,
            sequence_id=3,
            timestamp_s=10.134,
        ),
        valid,
    )

    assert (
        temporal_3.body.state
        == TrackingState.TRACKING
    )
    assert temporal_3.body.valid_for_control
    assert temporal_3.left_hand.valid_for_control
    assert temporal_3.right_hand.valid_for_control

    print(
        "H4 temporal initial acquisition gate: PASS"
    )

    temporal_4 = tracker.update(
        shifted_observation(
            missing_observation,
            sequence_id=4,
            timestamp_s=10.201,
        ),
        missing_validity,
    )

    assert (
        temporal_4.body.state
        == TrackingState.DROPOUT
    )
    assert not temporal_4.body.valid_for_control
    assert (
        temporal_4.body.age_since_last_valid_s
        is not None
    )
    assert (
        temporal_4.body.age_since_last_valid_s
        < temporal_config.dropout_timeout_s
    )

    print(
        "H4 temporal short-dropout handling: PASS"
    )

    temporal_5 = tracker.update(
        shifted_observation(
            missing_observation,
            sequence_id=5,
            timestamp_s=10.500,
        ),
        missing_validity,
    )

    assert (
        temporal_5.body.state
        == TrackingState.LOST
    )
    assert not temporal_5.body.valid_for_control

    print(
        "H4 temporal loss-timeout handling: PASS"
    )

    temporal_6 = tracker.update(
        shifted_observation(
            observation,
            sequence_id=6,
            timestamp_s=10.567,
        ),
        valid,
    )

    assert (
        temporal_6.body.state
        == TrackingState.ACQUIRING
    )
    assert not temporal_6.body.valid_for_control

    temporal_7 = tracker.update(
        shifted_observation(
            observation,
            sequence_id=7,
            timestamp_s=10.634,
        ),
        valid,
    )

    assert (
        temporal_7.body.state
        == TrackingState.ACQUIRING
    )
    assert not temporal_7.body.valid_for_control

    temporal_8 = tracker.update(
        shifted_observation(
            observation,
            sequence_id=8,
            timestamp_s=10.701,
        ),
        valid,
    )

    assert (
        temporal_8.body.state
        == TrackingState.TRACKING
    )
    assert temporal_8.body.valid_for_control

    print(
        "H4 temporal reacquisition gate: PASS"
    )

    try:
        tracker.update(
            shifted_observation(
                observation,
                sequence_id=8,
                timestamp_s=10.768,
            ),
            valid,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Duplicate sequence ID was accepted."
        )

    print(
        "H4 temporal ordering guard: PASS"
    )

    # ---------------------------------------------------------
    # H5 smoothing tests
    # ---------------------------------------------------------

    smoothing_config = LandmarkSmoothingConfig(
        time_constant_s=0.10,
    )

    smoothing_tracker = HumanTemporalTracker(
        temporal_config
    )

    smoother = HumanLandmarkSmoother(
        smoothing_config
    )

    # Frames 1 and 2 are still ACQUIRING, so no smoothed control
    # landmarks may be emitted.
    smooth_obs_1 = with_body_landmark_x(
        observation,
        landmark_index=15,
        x=0.50,
        sequence_id=1,
        timestamp_s=20.000,
    )

    smooth_valid_1 = evaluate_observation(
        smooth_obs_1,
        config,
    )

    smooth_track_1 = smoothing_tracker.update(
        smooth_obs_1,
        smooth_valid_1,
    )

    smooth_result_1 = smoother.update(
        smooth_obs_1,
        smooth_track_1,
    )

    assert (
        smooth_result_1.body.image_landmarks
        is None
    )

    smooth_obs_2 = with_body_landmark_x(
        observation,
        landmark_index=15,
        x=0.55,
        sequence_id=2,
        timestamp_s=20.067,
    )

    smooth_valid_2 = evaluate_observation(
        smooth_obs_2,
        config,
    )

    smooth_track_2 = smoothing_tracker.update(
        smooth_obs_2,
        smooth_valid_2,
    )

    smooth_result_2 = smoother.update(
        smooth_obs_2,
        smooth_track_2,
    )

    assert (
        smooth_result_2.body.image_landmarks
        is None
    )

    # Frame 3 reaches TRACKING and seeds the filter.
    smooth_obs_3 = with_body_landmark_x(
        observation,
        landmark_index=15,
        x=0.50,
        sequence_id=3,
        timestamp_s=20.134,
    )

    smooth_valid_3 = evaluate_observation(
        smooth_obs_3,
        config,
    )

    smooth_track_3 = smoothing_tracker.update(
        smooth_obs_3,
        smooth_valid_3,
    )

    smooth_result_3 = smoother.update(
        smooth_obs_3,
        smooth_track_3,
    )

    assert (
        smooth_result_3.body.image_landmarks
        is not None
    )

    seed_x = (
        smooth_result_3
        .body
        .image_landmarks
        .landmarks[15]
        .x
    )

    assert abs(seed_x - 0.50) < 1e-12

    # A sudden 0.10 raw jump must be reduced by the low-pass filter.
    smooth_obs_4 = with_body_landmark_x(
        observation,
        landmark_index=15,
        x=0.60,
        sequence_id=4,
        timestamp_s=20.201,
    )

    smooth_valid_4 = evaluate_observation(
        smooth_obs_4,
        config,
    )

    smooth_track_4 = smoothing_tracker.update(
        smooth_obs_4,
        smooth_valid_4,
    )

    smooth_result_4 = smoother.update(
        smooth_obs_4,
        smooth_track_4,
    )

    assert (
        smooth_result_4.body.image_landmarks
        is not None
    )

    filtered_x = (
        smooth_result_4
        .body
        .image_landmarks
        .landmarks[15]
        .x
    )

    assert 0.50 < filtered_x < 0.60

    raw_step = abs(
        0.60 - 0.50
    )

    filtered_step = abs(
        filtered_x - 0.50
    )

    assert filtered_step < raw_step

    print(
        "H5 valid-measurement smoothing: PASS"
    )

    # A dropout must emit no control-side smoothed body landmarks and
    # reset the filter state.
    smooth_missing_5 = shifted_observation(
        missing_observation,
        sequence_id=5,
        timestamp_s=20.268,
    )

    smooth_missing_validity_5 = (
        evaluate_observation(
            smooth_missing_5,
            config,
        )
    )

    smooth_track_5 = smoothing_tracker.update(
        smooth_missing_5,
        smooth_missing_validity_5,
    )

    smooth_result_5 = smoother.update(
        smooth_missing_5,
        smooth_track_5,
    )

    assert (
        smooth_result_5.body.image_landmarks
        is None
    )

    print(
        "H5 invalid-measurement suppression: PASS"
    )

    # Reacquisition again requires three frames. The filter is reset,
    # so the first control-valid result seeds directly from fresh data.
    for sequence_id, timestamp_s in (
        (6, 20.335),
        (7, 20.402),
        (8, 20.469),
    ):
        reacquired_observation = (
            with_body_landmark_x(
                observation,
                landmark_index=15,
                x=0.80,
                sequence_id=sequence_id,
                timestamp_s=timestamp_s,
            )
        )

        reacquired_validity = (
            evaluate_observation(
                reacquired_observation,
                config,
            )
        )

        reacquired_tracking = (
            smoothing_tracker.update(
                reacquired_observation,
                reacquired_validity,
            )
        )

        reacquired_smoothed = (
            smoother.update(
                reacquired_observation,
                reacquired_tracking,
            )
        )

    assert (
        reacquired_tracking.body.state
        == TrackingState.TRACKING
    )

    assert (
        reacquired_smoothed
        .body
        .image_landmarks
        is not None
    )

    reacquired_x = (
        reacquired_smoothed
        .body
        .image_landmarks
        .landmarks[15]
        .x
    )

    assert abs(reacquired_x - 0.80) < 1e-12

    print(
        "H5 reset-on-dropout/reacquisition: PASS"
    )

    try:
        LandmarkSmoothingConfig(
            time_constant_s=0.0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Invalid smoothing time constant was accepted."
        )

    print(
        "H5 smoothing configuration check: PASS"
    )

    # ---------------------------------------------------------
    # H6 derived body/hand geometry tests
    # ---------------------------------------------------------

    if observation.body_world_landmarks is None:
        raise AssertionError(
            "Synthetic observation has no body world landmarks."
        )

    if (
        observation.left_hands[0].world_landmarks
        is None
        or observation.right_hands[0].world_landmarks
        is None
    ):
        raise AssertionError(
            "Synthetic observation has no hand world landmarks."
        )

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
        17: (-0.04, 0.04, 0.00),
    }

    left_hand_world = landmark_set_with_positions(
        observation.left_hands[0].world_landmarks,
        hand_positions,
    )

    right_hand_world = landmark_set_with_positions(
        observation.right_hands[0].world_landmarks,
        hand_positions,
    )

    derived_input = SmoothedHumanLandmarks(
        frame_sequence_id=42,
        measurement_timestamp_s=30.000,
        body=SmoothedChannelLandmarks(
            image_landmarks=None,
            world_landmarks=body_world,
        ),
        left_hand=SmoothedChannelLandmarks(
            image_landmarks=None,
            world_landmarks=left_hand_world,
        ),
        right_hand=SmoothedChannelLandmarks(
            image_landmarks=None,
            world_landmarks=right_hand_world,
        ),
    )

    derived = derive_human_kinematics(
        derived_input,
        DerivedGeometryConfig(
            min_length_model_world=1e-5,
        ),
    )

    assert derived.body is not None
    assert derived.left_hand is not None
    assert derived.right_hand is not None

    assert not derived.body_reasons
    assert not derived.left_hand_reasons
    assert not derived.right_hand_reasons

    assert_close(
        derived.body.frame.right_axis_model_world.x,
        1.0,
    )

    assert_close(
        derived.body.frame.up_axis_model_world.y,
        1.0,
    )

    assert_close(
        derived.body.frame.normal_axis_model_world.z,
        1.0,
    )

    assert_close(
        derived.body.shoulder_width_model_world,
        0.40,
    )

    assert_close(
        derived.body.torso_height_model_world,
        0.40,
    )

    assert (
        derived
        .body
        .left_wrist_displacement_normalized_body
        .x
        < 0.0
    )

    assert (
        derived
        .body
        .right_wrist_displacement_normalized_body
        .x
        > 0.0
    )

    assert (
        derived
        .body
        .left_wrist_displacement_normalized_body
        .y
        < 0.0
    )

    assert (
        derived
        .body
        .right_wrist_displacement_normalized_body
        .y
        < 0.0
    )

    print(
        "H6 body-relative frame and arm geometry: PASS"
    )

    assert_close(
        derived.left_hand.palm_width_model_world,
        0.08,
    )

    assert_close(
        derived.left_hand.pinch_ratio,
        0.125,
    )

    assert (
        derived
        .left_hand
        .palm_normal_model_world
        .z
        > 0.99
    )

    assert_close(
        derived.right_hand.pinch_ratio,
        0.125,
    )

    print(
        "H6 normalized hand geometry: PASS"
    )

    degenerate_body_world = (
        landmark_set_with_positions(
            body_world,
            {
                12: (-0.20, 0.40, 0.00),
            },
        )
    )

    degenerate_input = SmoothedHumanLandmarks(
        frame_sequence_id=43,
        measurement_timestamp_s=30.067,
        body=SmoothedChannelLandmarks(
            image_landmarks=None,
            world_landmarks=degenerate_body_world,
        ),
        left_hand=SmoothedChannelLandmarks(
            image_landmarks=None,
            world_landmarks=None,
        ),
        right_hand=SmoothedChannelLandmarks(
            image_landmarks=None,
            world_landmarks=None,
        ),
    )

    degenerate = derive_human_kinematics(
        degenerate_input,
        DerivedGeometryConfig(),
    )

    assert degenerate.body is None

    assert (
        "body_shoulder_width_degenerate"
        in degenerate.body_reasons
    )

    assert degenerate.left_hand is None
    assert degenerate.right_hand is None

    print(
        "H6 degenerate/missing geometry rejection: PASS"
    )

    try:
        DerivedGeometryConfig(
            min_length_model_world=0.0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Invalid derived-geometry guard was accepted."
        )

    print(
        "H6 geometry configuration check: PASS"
    )

    # ---------------------------------------------------------
    # H8 robot-agnostic relative retargeting tests
    # ---------------------------------------------------------

    if derived.body is None:
        raise AssertionError(
            "H8 requires the valid H6 synthetic body geometry."
        )

    reference_robot_pose = RobotAgnosticPose(
        position=Vector3(
            x=1.0,
            y=2.0,
            z=3.0,
        ),
        orientation_xyzw=Quaternion(
            x=0.0,
            y=0.0,
            z=0.0,
            w=2.0,
        ),
    )

    mapping_config = RetargetingConfig(
        axis_mapping=AxisMapping(
            rows=(
                (0.0, 1.0, 0.0),
                (-1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        ),
        scale_robot_per_normalized_body=Vector3(
            x=2.0,
            y=3.0,
            z=4.0,
        ),
        deadband_normalized_body=Vector3(
            x=0.05,
            y=0.05,
            z=0.05,
        ),
        workspace_bounds=CartesianBounds(
            minimum=Vector3(
                x=-10.0,
                y=-10.0,
                z=-10.0,
            ),
            maximum=Vector3(
                x=10.0,
                y=10.0,
                z=10.0,
            ),
        ),
        max_cartesian_speed_robot_per_s=None,
    )

    retargeter = RelativeRetargeter(
        arm_side=ArmSide.LEFT,
        config=mapping_config,
    )

    reference = retargeter.activate(
        derived,
        reference_robot_pose,
    )

    assert reference.arm_side == ArmSide.LEFT

    assert_close(
        reference
        .robot_pose_reference
        .orientation_xyzw
        .w,
        1.0,
    )

    reference_wrist = (
        derived
        .body
        .left_wrist_displacement_normalized_body
    )

    moved_body = replace(
        derived.body,
        left_wrist_displacement_normalized_body=Vector3(
            x=reference_wrist.x + 0.20,
            y=reference_wrist.y - 0.10,
            z=reference_wrist.z + 0.10,
        ),
    )

    moved_human = replace(
        derived,
        frame_sequence_id=43,
        measurement_timestamp_s=30.100,
        body=moved_body,
    )

    mapped_result = retargeter.update(
        moved_human
    )

    assert mapped_result.valid
    assert mapped_result.active
    assert mapped_result.frame_sequence_id == 43
    assert_close(
        mapped_result.measurement_timestamp_s,
        30.100,
    )
    assert mapped_result.target_pose is not None
    assert not mapped_result.workspace_limited
    assert not mapped_result.speed_limited

    assert_close(
        mapped_result
        .target_pose
        .position
        .x,
        0.70,
    )

    assert_close(
        mapped_result
        .target_pose
        .position
        .y,
        1.60,
    )

    assert_close(
        mapped_result
        .target_pose
        .position
        .z,
        3.40,
    )

    assert_close(
        mapped_result
        .target_pose
        .orientation_xyzw
        .x,
        0.0,
    )

    assert_close(
        mapped_result
        .target_pose
        .orientation_xyzw
        .y,
        0.0,
    )

    assert_close(
        mapped_result
        .target_pose
        .orientation_xyzw
        .z,
        0.0,
    )

    assert_close(
        mapped_result
        .target_pose
        .orientation_xyzw
        .w,
        1.0,
    )

    print(
        "H8 relative mapping and fixed orientation: PASS"
    )

    deadband_retargeter = RelativeRetargeter(
        arm_side=ArmSide.LEFT,
        config=mapping_config,
    )

    deadband_retargeter.activate(
        derived,
        reference_robot_pose,
    )

    deadband_body = replace(
        derived.body,
        left_wrist_displacement_normalized_body=Vector3(
            x=reference_wrist.x + 0.02,
            y=reference_wrist.y - 0.03,
            z=reference_wrist.z + 0.01,
        ),
    )

    deadband_human = replace(
        derived,
        frame_sequence_id=43,
        measurement_timestamp_s=30.100,
        body=deadband_body,
    )

    deadband_result = (
        deadband_retargeter.update(
            deadband_human
        )
    )

    assert deadband_result.valid
    assert deadband_result.target_pose is not None

    assert_close(
        deadband_result.target_pose.position.x,
        1.0,
    )
    assert_close(
        deadband_result.target_pose.position.y,
        2.0,
    )
    assert_close(
        deadband_result.target_pose.position.z,
        3.0,
    )

    print(
        "H8 normalized deadband handling: PASS"
    )

    workspace_config = RetargetingConfig(
        axis_mapping=AxisMapping.identity(),
        scale_robot_per_normalized_body=Vector3(
            x=5.0,
            y=5.0,
            z=5.0,
        ),
        deadband_normalized_body=Vector3(
            x=0.0,
            y=0.0,
            z=0.0,
        ),
        workspace_bounds=CartesianBounds(
            minimum=Vector3(
                x=0.0,
                y=0.0,
                z=0.0,
            ),
            maximum=Vector3(
                x=1.1,
                y=2.1,
                z=3.1,
            ),
        ),
        max_cartesian_speed_robot_per_s=None,
    )

    workspace_retargeter = RelativeRetargeter(
        arm_side=ArmSide.LEFT,
        config=workspace_config,
    )

    workspace_retargeter.activate(
        derived,
        reference_robot_pose,
    )

    outside_body = replace(
        derived.body,
        left_wrist_displacement_normalized_body=Vector3(
            x=reference_wrist.x + 1.0,
            y=reference_wrist.y + 1.0,
            z=reference_wrist.z + 1.0,
        ),
    )

    outside_human = replace(
        derived,
        frame_sequence_id=43,
        measurement_timestamp_s=30.100,
        body=outside_body,
    )

    workspace_result = (
        workspace_retargeter.update(
            outside_human
        )
    )

    assert workspace_result.valid
    assert workspace_result.workspace_limited
    assert workspace_result.target_pose is not None

    assert_close(
        workspace_result.target_pose.position.x,
        1.1,
    )
    assert_close(
        workspace_result.target_pose.position.y,
        2.1,
    )
    assert_close(
        workspace_result.target_pose.position.z,
        3.1,
    )

    print(
        "H8 workspace-limit handling: PASS"
    )

    speed_config = RetargetingConfig(
        axis_mapping=AxisMapping.identity(),
        scale_robot_per_normalized_body=Vector3(
            x=10.0,
            y=10.0,
            z=10.0,
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
        max_cartesian_speed_robot_per_s=1.0,
    )

    speed_retargeter = RelativeRetargeter(
        arm_side=ArmSide.LEFT,
        config=speed_config,
    )

    speed_retargeter.activate(
        derived,
        reference_robot_pose,
    )

    fast_body = replace(
        derived.body,
        left_wrist_displacement_normalized_body=Vector3(
            x=reference_wrist.x + 1.0,
            y=reference_wrist.y,
            z=reference_wrist.z,
        ),
    )

    fast_human = replace(
        derived,
        frame_sequence_id=43,
        measurement_timestamp_s=30.100,
        body=fast_body,
    )

    speed_result = (
        speed_retargeter.update(
            fast_human
        )
    )

    assert speed_result.valid
    assert speed_result.speed_limited
    assert speed_result.target_pose is not None

    speed_step_x = (
        speed_result.target_pose.position.x
        - reference_robot_pose.position.x
    )

    assert_close(
        speed_step_x,
        0.10,
        tolerance=1e-9,
    )

    print(
        "H8 Cartesian speed-limit handling: PASS"
    )

    invalid_human = replace(
        derived,
        frame_sequence_id=43,
        measurement_timestamp_s=30.100,
        body=None,
    )

    invalid_retargeter = RelativeRetargeter(
        arm_side=ArmSide.RIGHT,
        config=mapping_config,
    )

    inactive_result = invalid_retargeter.update(
        invalid_human
    )

    assert not inactive_result.valid
    assert not inactive_result.active
    assert inactive_result.target_pose is None

    assert (
        "retargeting_inactive"
        in inactive_result.reasons
    )

    invalid_retargeter.activate(
        derived,
        reference_robot_pose,
    )

    missing_result = invalid_retargeter.update(
        invalid_human
    )

    assert not missing_result.valid
    assert missing_result.active
    assert missing_result.target_pose is None

    assert (
        "body_derived_kinematics_unavailable"
        in missing_result.reasons
    )

    print(
        "H8 fail-closed invalid-input handling: PASS"
    )

    invalid_order_retargeter = RelativeRetargeter(
        arm_side=ArmSide.RIGHT,
        config=mapping_config,
    )

    invalid_order_retargeter.activate(
        derived,
        reference_robot_pose,
    )

    first_invalid_order_result = (
        invalid_order_retargeter.update(
            invalid_human
        )
    )

    assert not first_invalid_order_result.valid
    assert first_invalid_order_result.active

    try:
        invalid_order_retargeter.update(
            invalid_human
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "H8 accepted the same invalid frame twice."
        )

    print(
        "H8 invalid-frame ordering guard: PASS"
    )

    ordering_retargeter = RelativeRetargeter(
        arm_side=ArmSide.LEFT,
        config=mapping_config,
    )

    ordering_retargeter.activate(
        derived,
        reference_robot_pose,
    )

    try:
        ordering_retargeter.update(
            replace(
                derived,
                frame_sequence_id=42,
                measurement_timestamp_s=30.000,
            )
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "H8 accepted a non-increasing timestamp."
        )

    print(
        "H8 timestamp ordering guard: PASS"
    )

    try:
        AxisMapping(
            rows=(
                (1.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Invalid axis mapping was accepted."
        )

    try:
        RetargetingConfig(
            axis_mapping=AxisMapping.identity(),
            scale_robot_per_normalized_body=Vector3(
                x=1.0,
                y=1.0,
                z=1.0,
            ),
            deadband_normalized_body=Vector3(
                x=-0.1,
                y=0.0,
                z=0.0,
            ),
            workspace_bounds=CartesianBounds(
                minimum=Vector3(
                    x=-1.0,
                    y=-1.0,
                    z=-1.0,
                ),
                maximum=Vector3(
                    x=1.0,
                    y=1.0,
                    z=1.0,
                ),
            ),
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Invalid retargeting configuration was accepted."
        )

    print(
        "H8 configuration guards: PASS"
    )

    print("-" * 60)
    print("H3 RESULT: PASS")
    print("H4 RESULT: PASS")
    print("H4 TEMPORAL RESULT: PASS")
    print("H5 RESULT: PASS")
    print("H6 RESULT: PASS")
    print("H8 RESULT: PASS")
    print(
        "Robot-independent relative Cartesian retargeting passed "
        "synthetic mapping, deadband, workspace, rate-limit, "
        "invalid-input and ordering checks."
    )


if __name__ == "__main__":
    main()
