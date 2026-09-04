import platform
import sys

def report_api(module, name):
    available = hasattr(module, name)
    print(f"{name}: {'AVAILABLE' if available else 'NOT AVAILABLE'}")
    return available


def main():
    print("=" * 60)
    print("H0 - Human Tracking Environment Check")
    print("=" * 60)

    print(f"Python executable: {sys.executable}")
    print(f"Python version:    {sys.version.split()[0]}")
    print(f"Platform:          {platform.platform()}")
    print()

    try:
        import mediapipe as mp

    except Exception as exc:
        print("FAIL: Could not import mediapipe.")
        print(f"{type(exc).__name__}: {exc}")
        raise SystemExit(1)

    print(f"MediaPipe version: {mp.__version__}")

    try:
        import cv2
        print(f"OpenCV version:    {cv2.__version__}")

    except Exception as exc:
        print("OpenCV import:     NOT AVAILABLE")
        print(f"Reason: {type(exc).__name__}: {exc}")

    print()

    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

    except Exception as exc:
        print("FAIL: MediaPipe Tasks API could not be imported.")
        print(f"{type(exc).__name__}: {exc}")
        raise SystemExit(1)

    print("MediaPipe Tasks imports: AVAILABLE")
    print()

    required_apis = {
        "BaseOptions": hasattr(mp_python, "BaseOptions"),
        "PoseLandmarker": report_api(vision, "PoseLandmarker"),
        "PoseLandmarkerOptions": report_api(vision, "PoseLandmarkerOptions"),
        "HandLandmarker": report_api(vision, "HandLandmarker"),
        "HandLandmarkerOptions": report_api(vision, "HandLandmarkerOptions"),
        "HolisticLandmarker": report_api(vision, "HolisticLandmarker"),
        "HolisticLandmarkerOptions": report_api(vision, "HolisticLandmarkerOptions"),
        "RunningMode": report_api(vision, "RunningMode")
    }

    print(
        f"BaseOptions: "
        f"{'AVAILABLE' if required_apis['BaseOptions'] else 'NOT AVAILABLE'}"
    )

    print()
    print("-" * 60)

    essential = (
        required_apis["BaseOptions"]
        and required_apis["PoseLandmarker"]
        and required_apis["PoseLandmarkerOptions"]
        and required_apis["HandLandmarker"]
        and required_apis["HandLandmarkerOptions"]
        and required_apis["RunningMode"]
    )

    if essential:
        print("H0 RESULT: PASS")
        print("Pose + Hand MediaPipe Tasks APIs are available in this environment.")

        if required_apis["HolisticLandmarker"] and required_apis["HolisticLandmarkerOptions"]:
            print("Holistic Landmarker API is also available.")

        else:
            print("Holistic Landmarker API is not exposed here; Pose + Hand remain available.")

        raise SystemExit(0)

    print("H0 RESULT: FAIL")
    print("One or more required MediaPipe Tasks APIs are unavailable.")
    raise SystemExit(1)

if __name__ == "__main__":
    main()