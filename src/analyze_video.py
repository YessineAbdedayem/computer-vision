"""Analyse hors ligne d'une vidéo et export des angles et répétitions."""

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapipe as mp
import pandas as pd

if __package__:
    from .angles import calculate_angle
    from .rep_counter import RepCounter
else:
    from angles import calculate_angle
    from rep_counter import RepCounter


VISIBILITY_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse hors ligne des squats d'une vidéo.")
    parser.add_argument("video", help="Chemin de la vidéo à analyser.")
    parser.add_argument("--side", choices=("left", "right"), default="right")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.is_file():
        print(f"Fichier vidéo introuvable : {video_path}")
        return 1

    pose_module = getattr(getattr(mp, "solutions", None), "pose", None)
    if pose_module is None:
        print(
            "Cette version de MediaPipe ne fournit pas mp.solutions.pose. "
            "Installez les dépendances du projet avec Python 3.12."
        )
        return 1

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        print(f"Impossible d'ouvrir la vidéo : {video_path}")
        return 1

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
    counter = RepCounter()
    records: list[dict[str, int | float | str]] = []
    repetition_points: list[tuple[float, float]] = []
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        capture.release()
        print("La fréquence d'images de la vidéo est invalide.")
        return 1

    try:
        with pose_module.Pose(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            frame_index = 0
            while True:
                success, frame = capture.read()
                if not success:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)
                timestamp = frame_index / fps
                angle = float("nan")

                if results.pose_landmarks:
                    hip_id, knee_id, ankle_id = side_indices[args.side]
                    landmarks = results.pose_landmarks.landmark
                    selected = [landmarks[index.value] for index in (hip_id, knee_id, ankle_id)]
                    if all(point.visibility >= VISIBILITY_THRESHOLD for point in selected):
                        height, width = frame.shape[:2]
                        points = [
                            (point.x * width, point.y * height)
                            for point in selected
                        ]
                        angle = calculate_angle(*points)
                        if counter.update(angle):
                            repetition_points.append((timestamp, angle))

                records.append(
                    {
                        "frame": frame_index,
                        "temps": timestamp,
                        "angle_genou": angle,
                        "etat": counter.state_label,
                        "nb_reps": counter.repetitions,
                    }
                )
                frame_index += 1
    except (cv2.error, RuntimeError) as error:
        print(f"Erreur pendant l'analyse : {error}")
        return 1
    finally:
        capture.release()

    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    csv_path = output_dir / f"{stem}_analysis.csv"
    plot_path = output_dir / f"{stem}_angles.png"
    data = pd.DataFrame(
        records,
        columns=("frame", "temps", "angle_genou", "etat", "nb_reps"),
    )
    data.to_csv(csv_path, index=False)

    valid_angles = data["angle_genou"].notna()
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(
        data.loc[valid_angles, "temps"],
        data.loc[valid_angles, "angle_genou"],
        label="Angle du genou",
    )
    if repetition_points:
        rep_times, rep_angles = zip(*repetition_points)
        axis.scatter(rep_times, rep_angles, color="red", label="Répétition détectée")
    axis.set(
        title="Évolution de l'angle du genou",
        xlabel="Temps (secondes)",
        ylabel="Angle (degrés)",
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_path)
    plt.close(figure)

    print(f"Répétitions détectées : {counter.repetitions}")
    print(f"CSV : {csv_path}")
    print(f"Graphique : {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
