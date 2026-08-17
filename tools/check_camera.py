from __future__ import annotations

import argparse
import time

import cv2

from human_tracking.acquisition import CameraAcquisition


WINDOW_NAME = "H1 - Human Tracking Camera Check"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Verify the timestamped camera acquisition boundary "
            "for Part 2."
        )
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera index. Default: 0",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("H1 - Camera / Acquisition Boundary Check")
    print("=" * 60)
    print(f"Requested camera index: {args.camera}")
    print()

    try:
        acquisition = CameraAcquisition(
            camera_index=args.camera
        )
        acquisition.open()
    except Exception as exc:
        print("H1 RESULT: FAIL")
        print(f"{type(exc).__name__}: {exc}")
        raise SystemExit(1)

    try:
        properties = acquisition.get_reported_properties()

        print("Camera opened successfully.")
        print(
            "Camera-reported resolution: "
            f"{properties['width_px']:.0f} x "
            f"{properties['height_px']:.0f}"
        )
        print(
            "Camera-reported FPS: "
            f"{properties['fps']:.2f}"
        )
        print(
            "Measurement timestamp source: "
            "time.perf_counter() once per successful image"
        )
        print()
        print("Press Q in the camera window to finish.")
        print("-" * 60)

        first_timestamp_s = None
        previous_timestamp_s = None

        frame_count = 0
        timestamp_order_errors = 0

        start_wall_s = time.perf_counter()

        while True:
            try:
                frame = acquisition.read()
            except Exception as exc:
                print()
                print("Frame acquisition failed.")
                print(f"{type(exc).__name__}: {exc}")
                raise SystemExit(1)

            if first_timestamp_s is None:
                first_timestamp_s = (
                    frame.measurement_timestamp_s
                )

            if previous_timestamp_s is not None:
                if (
                    frame.measurement_timestamp_s
                    <= previous_timestamp_s
                ):
                    timestamp_order_errors += 1

            previous_timestamp_s = (
                frame.measurement_timestamp_s
            )

            frame_count += 1

            display = frame.image_bgr.copy()

            elapsed_s = (
                frame.measurement_timestamp_s
                - first_timestamp_s
            )

            measured_fps = (
                frame_count / elapsed_s
                if elapsed_s > 0.0
                else 0.0
            )

            cv2.putText(
                display,
                f"Frame: {frame.sequence_id}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                (
                    f"Timestamp: "
                    f"{frame.measurement_timestamp_s:.6f} s"
                ),
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                (
                    f"Image: "
                    f"{frame.image_width_px} x "
                    f"{frame.image_height_px}"
                ),
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                f"Measured FPS: {measured_fps:.1f}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(
                WINDOW_NAME,
                display,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                break

        end_wall_s = time.perf_counter()
        run_duration_s = end_wall_s - start_wall_s

        print()
        print("-" * 60)
        print(f"Frames acquired: {frame_count}")
        print(
            f"Run duration: {run_duration_s:.3f} s"
        )

        if run_duration_s > 0:
            print(
                "Average acquisition/display rate: "
                f"{frame_count / run_duration_s:.2f} FPS"
            )

        print(
            "Timestamp ordering errors: "
            f"{timestamp_order_errors}"
        )

        if (
            frame_count > 0
            and timestamp_order_errors == 0
        ):
            print("H1 RESULT: PASS")
            print(
                "Camera acquisition and monotonic "
                "one-image timestamping verified."
            )
        else:
            print("H1 RESULT: FAIL")
            print(
                "No usable frame sequence was verified."
            )
            raise SystemExit(1)

    finally:
        acquisition.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()