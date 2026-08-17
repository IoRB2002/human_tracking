from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from human_tracking.acquisition import CameraFrame


@dataclass(frozen=True)
class MediaPipeTrackingOutput:
    """
    Raw MediaPipe output associated with exactly one acquired camera image.

    This remains backend-specific. A later milestone will convert this into
    the transport-independent HumanObservation representation.
    """

    frame_sequence_id: int
    measurement_timestamp_s: float
    mediapipe_timestamp_ms: int

    pose_result: Any
    hand_result: Any

    processing_duration_s: float


class MediaPipeTrackingBackend:
    """
    Synchronous MediaPipe Pose + Hand backend.

    The backend receives CameraFrame objects from the acquisition boundary.
    It never owns the camera and never creates a new measurement timestamp.

    MediaPipe VIDEO mode receives a millisecond form of the original
    monotonic camera timestamp.
    """

    def __init__(
        self,
        pose_model_path: str | Path,
        hand_model_path: str | Path,
    ):
        self.pose_model_path = Path(
            pose_model_path
        ).expanduser().resolve()

        self.hand_model_path = Path(
            hand_model_path
        ).expanduser().resolve()

        if not self.pose_model_path.is_file():
            raise FileNotFoundError(
                f"Pose model not found: {self.pose_model_path}"
            )

        if not self.hand_model_path.is_file():
            raise FileNotFoundError(
                f"Hand model not found: {self.hand_model_path}"
            )

        pose_options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(
                    self.pose_model_path
                )
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )

        hand_options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(
                    self.hand_model_path
                )
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
        )

        self._pose_landmarker = (
            vision.PoseLandmarker.create_from_options(
                pose_options
            )
        )

        try:
            self._hand_landmarker = (
                vision.HandLandmarker.create_from_options(
                    hand_options
                )
            )
        except Exception:
            self._pose_landmarker.close()
            raise

        self._last_timestamp_ms: int | None = None
        self._last_sequence_id: int | None = None

    def process(
        self,
        frame: CameraFrame,
    ) -> MediaPipeTrackingOutput:
        """
        Process one CameraFrame.

        The original measurement_timestamp_s is preserved unchanged.
        """

        if not np.isfinite(
            frame.measurement_timestamp_s
        ):
            raise ValueError(
                "Camera frame timestamp must be finite."
            )

        if frame.measurement_timestamp_s < 0.0:
            raise ValueError(
                "Camera frame timestamp must be non-negative."
            )

        if self._last_sequence_id is not None:
            if frame.sequence_id <= self._last_sequence_id:
                raise ValueError(
                    "Camera frame sequence IDs must increase."
                )

        # MediaPipe VIDEO mode uses integer milliseconds.
        timestamp_ms = int(
            frame.measurement_timestamp_s * 1000.0
        )

        if self._last_timestamp_ms is not None:
            if timestamp_ms <= self._last_timestamp_ms:
                raise ValueError(
                    "MediaPipe VIDEO timestamps must be "
                    "strictly increasing."
                )

        rgb_image = cv2.cvtColor(
            frame.image_bgr,
            cv2.COLOR_BGR2RGB,
        )

        rgb_image = np.ascontiguousarray(
            rgb_image
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image,
        )

        processing_start_s = time.perf_counter()

        pose_result = (
            self._pose_landmarker.detect_for_video(
                mp_image,
                timestamp_ms,
            )
        )

        hand_result = (
            self._hand_landmarker.detect_for_video(
                mp_image,
                timestamp_ms,
            )
        )

        processing_duration_s = (
            time.perf_counter()
            - processing_start_s
        )

        # Advance backend ordering state only after both tasks succeeded.
        self._last_timestamp_ms = timestamp_ms
        self._last_sequence_id = frame.sequence_id

        return MediaPipeTrackingOutput(
            frame_sequence_id=frame.sequence_id,
            measurement_timestamp_s=(
                frame.measurement_timestamp_s
            ),
            mediapipe_timestamp_ms=timestamp_ms,
            pose_result=pose_result,
            hand_result=hand_result,
            processing_duration_s=(
                processing_duration_s
            ),
        )

    def close(self) -> None:
        if self._pose_landmarker is not None:
            self._pose_landmarker.close()
            self._pose_landmarker = None

        if self._hand_landmarker is not None:
            self._hand_landmarker.close()
            self._hand_landmarker = None

    def __enter__(
        self,
    ) -> "MediaPipeTrackingBackend":
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()