from __future__ import annotations

from dataclasses import dataclass
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraFrame:
    """
    One image acquired from the human-tracking camera.

    Exactly one measurement timestamp belongs to the complete image.
    Every body or hand observation later derived from this image must
    preserve this timestamp.

    The timestamp is sampled from time.perf_counter(), so it is a local
    monotonic measurement timestamp and is intentionally independent of
    ROS 2 transport or wall-clock time.
    """

    sequence_id: int
    measurement_timestamp_s: float

    image_bgr: np.ndarray

    image_width_px: int
    image_height_px: int


class CameraAcquisition:
    """
    OpenCV camera acquisition boundary for the human-tracking core.

    Camera capture is deliberately kept separate from MediaPipe and from
    later ROS 2 integration.
    """

    def __init__(self, camera_index: int = 0):
        if camera_index < 0:
            raise ValueError("camera_index must be non-negative.")

        self.camera_index = camera_index
        self._camera: cv2.VideoCapture | None = None
        self._sequence_id = 0

    @property
    def is_open(self) -> bool:
        return (
            self._camera is not None
            and self._camera.isOpened()
        )

    def open(self) -> None:
        if self.is_open:
            return

        # This mirrors the Windows acquisition approach already verified
        # during Part 1.
        camera = cv2.VideoCapture(
            self.camera_index,
            cv2.CAP_DSHOW,
        )

        if not camera.isOpened():
            camera.release()

            # Fall back to OpenCV's default backend.
            camera = cv2.VideoCapture(
                self.camera_index,
            )

        if not camera.isOpened():
            camera.release()
            raise RuntimeError(
                f"Could not open camera index {self.camera_index}."
            )

        self._camera = camera
        self._sequence_id = 0

    def read(self) -> CameraFrame:
        if not self.is_open:
            raise RuntimeError(
                "Camera is not open. Call open() before read()."
            )

        assert self._camera is not None

        success, image_bgr = self._camera.read()

        if not success or image_bgr is None:
            raise RuntimeError(
                "Camera opened successfully but a frame "
                "could not be retrieved."
            )

        # One timestamp for this complete successfully acquired image.
        measurement_timestamp_s = time.perf_counter()

        if image_bgr.ndim != 3:
            raise RuntimeError(
                "Camera returned an image with an unexpected shape: "
                f"{image_bgr.shape!r}"
            )

        image_height_px, image_width_px = image_bgr.shape[:2]

        if image_width_px <= 0 or image_height_px <= 0:
            raise RuntimeError(
                "Camera returned invalid image dimensions."
            )

        frame = CameraFrame(
            sequence_id=self._sequence_id,
            measurement_timestamp_s=measurement_timestamp_s,
            image_bgr=image_bgr,
            image_width_px=int(image_width_px),
            image_height_px=int(image_height_px),
        )

        self._sequence_id += 1

        return frame

    def get_reported_properties(self) -> dict[str, float]:
        if not self.is_open:
            raise RuntimeError(
                "Camera is not open."
            )

        assert self._camera is not None

        return {
            "width_px": self._camera.get(
                cv2.CAP_PROP_FRAME_WIDTH
            ),
            "height_px": self._camera.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            ),
            "fps": self._camera.get(
                cv2.CAP_PROP_FPS
            ),
        }

    def close(self) -> None:
        if self._camera is not None:
            self._camera.release()
            self._camera = None

    def __enter__(self) -> "CameraAcquisition":
        self.open()
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()