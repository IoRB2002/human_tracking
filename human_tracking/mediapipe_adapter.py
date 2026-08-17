from __future__ import annotations

import math

import mediapipe as mp

from human_tracking.acquisition import CameraFrame
from human_tracking.mediapipe_backend import (
    MediaPipeTrackingOutput,
)
from human_tracking.observation import (
    CoordinateSpace,
    HandObservation,
    HumanObservation,
    Landmark,
    LandmarkSet,
)


POSE_LANDMARK_NAMES = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)


HAND_LANDMARK_NAMES = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_finger_mcp",
    "index_finger_pip",
    "index_finger_dip",
    "index_finger_tip",
    "middle_finger_mcp",
    "middle_finger_pip",
    "middle_finger_dip",
    "middle_finger_tip",
    "ring_finger_mcp",
    "ring_finger_pip",
    "ring_finger_dip",
    "ring_finger_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)


class MediaPipeObservationAdapter:
    """
    Convert backend-specific MediaPipe results into HumanObservation.
    """

    def convert(
        self,
        frame: CameraFrame,
        output: MediaPipeTrackingOutput,
    ) -> HumanObservation:
        if (
            frame.sequence_id
            != output.frame_sequence_id
        ):
            raise ValueError(
                "Frame sequence ID does not match "
                "the MediaPipe output."
            )

        if (
            frame.measurement_timestamp_s
            != output.measurement_timestamp_s
        ):
            raise ValueError(
                "Frame measurement timestamp does not match "
                "the MediaPipe output."
            )

        body_image_landmarks = None
        body_world_landmarks = None

        pose_sets = (
            output.pose_result.pose_landmarks
        )

        pose_world_sets = getattr(
            output.pose_result,
            "pose_world_landmarks",
            [],
        )

        if len(pose_sets) > 1:
            raise ValueError(
                "Part 2 currently expects at most one "
                "operator pose."
            )

        if pose_sets:
            body_image_landmarks = (
                self._convert_landmark_set(
                    raw_landmarks=pose_sets[0],
                    names=POSE_LANDMARK_NAMES,
                    coordinate_space=(
                        CoordinateSpace.NORMALIZED_IMAGE
                    ),
                )
            )

            if pose_world_sets:
                body_world_landmarks = (
                    self._convert_landmark_set(
                        raw_landmarks=(
                            pose_world_sets[0]
                        ),
                        names=POSE_LANDMARK_NAMES,
                        coordinate_space=(
                            CoordinateSpace.MEDIAPIPE_WORLD
                        ),
                    )
                )

        left_hands: list[HandObservation] = []
        right_hands: list[HandObservation] = []
        unknown_hands: list[HandObservation] = []

        hand_sets = (
            output.hand_result.hand_landmarks
        )

        hand_world_sets = getattr(
            output.hand_result,
            "hand_world_landmarks",
            [],
        )

        handedness_sets = getattr(
            output.hand_result,
            "handedness",
            [],
        )

        for hand_index, raw_hand in enumerate(
            hand_sets
        ):
            handedness, score = (
                self._extract_handedness(
                    handedness_sets,
                    hand_index,
                )
            )

            image_landmarks = (
                self._convert_landmark_set(
                    raw_landmarks=raw_hand,
                    names=HAND_LANDMARK_NAMES,
                    coordinate_space=(
                        CoordinateSpace.NORMALIZED_IMAGE
                    ),
                )
            )

            world_landmarks = None

            if hand_index < len(
                hand_world_sets
            ):
                world_landmarks = (
                    self._convert_landmark_set(
                        raw_landmarks=(
                            hand_world_sets[
                                hand_index
                            ]
                        ),
                        names=(
                            HAND_LANDMARK_NAMES
                        ),
                        coordinate_space=(
                            CoordinateSpace.MEDIAPIPE_WORLD
                        ),
                    )
                )

            hand = HandObservation(
                handedness=handedness,
                handedness_score=score,
                image_landmarks=image_landmarks,
                world_landmarks=world_landmarks,
            )

            normalized_handedness = (
                handedness.strip().lower()
            )

            if normalized_handedness == "left":
                left_hands.append(hand)
            elif normalized_handedness == "right":
                right_hands.append(hand)
            else:
                unknown_hands.append(hand)

        return HumanObservation(
            frame_sequence_id=(
                frame.sequence_id
            ),
            measurement_timestamp_s=(
                frame.measurement_timestamp_s
            ),
            image_width_px=(
                frame.image_width_px
            ),
            image_height_px=(
                frame.image_height_px
            ),
            body_image_landmarks=(
                body_image_landmarks
            ),
            body_world_landmarks=(
                body_world_landmarks
            ),
            left_hands=tuple(
                left_hands
            ),
            right_hands=tuple(
                right_hands
            ),
            unknown_hands=tuple(
                unknown_hands
            ),
            backend_name="MediaPipe Tasks",
            backend_version=mp.__version__,
        )

    @staticmethod
    def _convert_landmark_set(
        raw_landmarks,
        names: tuple[str, ...],
        coordinate_space: CoordinateSpace,
    ) -> LandmarkSet:
        if len(raw_landmarks) != len(names):
            raise ValueError(
                "Unexpected landmark count: "
                f"expected {len(names)}, "
                f"received {len(raw_landmarks)}."
            )

        landmarks = []

        for index, (
            raw_landmark,
            name,
        ) in enumerate(
            zip(raw_landmarks, names)
        ):
            x = float(raw_landmark.x)
            y = float(raw_landmark.y)
            z = float(raw_landmark.z)

            if not all(
                math.isfinite(value)
                for value in (x, y, z)
            ):
                raise ValueError(
                    f"Landmark {index} contains "
                    "non-finite coordinates."
                )

            visibility = (
                MediaPipeObservationAdapter
                ._optional_finite_float(
                    getattr(
                        raw_landmark,
                        "visibility",
                        None,
                    )
                )
            )

            presence = (
                MediaPipeObservationAdapter
                ._optional_finite_float(
                    getattr(
                        raw_landmark,
                        "presence",
                        None,
                    )
                )
            )

            landmarks.append(
                Landmark(
                    index=index,
                    name=name,
                    x=x,
                    y=y,
                    z=z,
                    visibility=visibility,
                    presence=presence,
                )
            )

        return LandmarkSet(
            coordinate_space=coordinate_space,
            landmarks=tuple(landmarks),
        )

    @staticmethod
    def _extract_handedness(
        handedness_sets,
        hand_index: int,
    ) -> tuple[str, float | None]:
        if hand_index >= len(
            handedness_sets
        ):
            return "Unknown", None

        categories = (
            handedness_sets[hand_index]
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

        score = (
            MediaPipeObservationAdapter
            ._optional_finite_float(
                getattr(
                    category,
                    "score",
                    None,
                )
            )
        )

        return str(name), score

    @staticmethod
    def _optional_finite_float(
        value,
    ) -> float | None:
        if value is None:
            return None

        result = float(value)

        if not math.isfinite(result):
            return None

        return result