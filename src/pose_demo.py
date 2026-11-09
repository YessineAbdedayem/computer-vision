"""Démo webcam/vidéo du coach de fitness."""

import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp

if __package__:
    from .angles import calculate_angle
    from .rep_counter import RepCounter
else:
    from angles import calculate_angle
    from rep_counter import RepCounter


VISIBILITY_THRESHOLD = 0.5


def parse_source(source: str) -> int | str:
    """Interprète les nombres comme des index de webcam, sinon comme des chemins."""
    if source.isdigit():
        return int(source)
    if not Path(source).is_file():
        raise FileNotFoundError(f"Fichier vidéo introuvable : {source}")
    return source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compteur de squats avec MediaPipe Pose.")
    parser.add_argument("--source", default="0", help="Index webcam (0) ou chemin vidéo.")
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--save", action="store_true", help="Enregistre outputs/demo.mp4.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        source = parse_source(args.source)
    except FileNotFoundError as error:
        print(error)
        return 1

    pose_module = getattr(getattr(mp, "solutions", None), "pose", None)
    if pose_module is None:
        print(
            "Cette version de MediaPipe ne fournit pas mp.solutions.pose. "
            "Installez les dépendances du projet avec Python 3.12."
        )
        return 1

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        print(f"Impossible d'ouvrir la source vidéo : {args.source}")
        return 1

    drawing = mp.solutions.drawing_utils
    side_indices = {
        "left": (
            pose_module.PoseLandmark.LEFT_HIP,
            pose_module.PoseLandmark.LEFT_KNEE,
            pose_module.PoseLandmark.LEFT_ANKLE,
        ),
        "right": (
            pose_module.PoseLandmark.RIGHT_HIP,
            pose_module.PoseLandmark.RIGHT_KNEE,
            pose_module.PoseLandmark.RIGHT_ANKLE,
        ),
    }
    rep_counter = RepCounter()
    writer = None
    previous_time = time.perf_counter()
    fps = 0.0

    try:
        with pose_module.Pose(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            while True:
                success, frame = capture.read()
                if not success:
                    break

                frame = cv2.flip(frame, 1) if isinstance(source, int) else frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                results = pose.process(rgb_frame)
                rgb_frame.flags.writeable = True

                message = None
                if not results.pose_landmarks:
                    message = "Corps non detecte"
                else:
                    drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        pose_module.POSE_CONNECTIONS,
                    )
                    hip_id, knee_id, ankle_id = side_indices[args.side]
                    landmarks = results.pose_landmarks.landmark
                    selected = [landmarks[index.value] for index in (hip_id, knee_id, ankle_id)]

                    if any(point.visibility < VISIBILITY_THRESHOLD for point in selected):
                        message = "Visibilite des points trop faible"
                    else:
                        height, width = frame.shape[:2]
                        points = [
                            (point.x * width, point.y * height)
                            for point in selected
                        ]
                        angle = calculate_angle(*points)
                        rep_counter.update(angle)

                        cv2.putText(
                            frame,
                            f"Angle genou: {rep_counter.smoothed_angle:.1f} deg",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                        )
                        cv2.putText(
                            frame,
                            f"Etat: {rep_counter.state_label}",
                            (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2,
                        )
                        cv2.putText(
                            frame,
                            f"Reps: {rep_counter.repetitions}",
                            (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )

                current_time = time.perf_counter()
                elapsed = current_time - previous_time
                if elapsed > 0:
                    fps = 1.0 / elapsed
                previous_time = current_time
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )
                if message:
                    cv2.putText(
                        frame,
                        message,
                        (20, 185),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

                if args.save and writer is None:
                    output_dir = Path(__file__).resolve().parents[1] / "outputs"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    source_fps = capture.get(cv2.CAP_PROP_FPS)
                    output_fps = source_fps if source_fps > 0 else 20.0
                    height, width = frame.shape[:2]
                    output_path = output_dir / "demo.mp4"
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        output_fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Impossible de créer {output_path}.")

                if writer is not None:
                    writer.write(frame)

                cv2.imshow("Coach de fitness - Squats", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except (cv2.error, RuntimeError) as error:
        print(f"Erreur pendant le traitement : {error}")
        return 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
