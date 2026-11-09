# Coach de fitness intelligent

Prototype de comptage de répétitions de squats à partir d'une vidéo, avec
MediaPipe Pose. Le genou est suivi de profil et une répétition est comptée
lorsque l'utilisateur passe de la position basse à la position haute.

## Installation

Depuis la racine du projet, créez un environnement virtuel et installez les
dépendances avec Python 3.12 (MediaPipe 0.10.21 ne fournit pas de paquet pour
Python 3.13).

**Windows (PowerShell)**

Si Python 3.12 n'est pas installé, installez-le puis ouvrez un nouveau
terminal PowerShell :

```powershell
winget install --id Python.Python.3.12 --exact
py -0p
```

Vérifiez que Python 3.12 apparaît dans la liste avant de créer l'environnement :

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import mediapipe as mp; print(mp.__version__, hasattr(mp, 'solutions'))"
python src/pose_demo.py --source 0
```

**macOS / Linux**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Démo en direct

Depuis la racine du projet, lancez la webcam (remplacez `0` par le chemin d'une
vidéo pour analyser un fichier) :

```bash
python src/pose_demo.py --source 0
python src/pose_demo.py --source chemin/vers/video.mp4 --side left
python src/pose_demo.py --source 0 --save
```

`--side` accepte `left` ou `right` (défaut : `right`). `--save` écrit la vidéo
annotée dans `outputs/demo.mp4`. Appuyez sur `q` pour quitter.

## Analyse hors ligne

L'analyse traite une vidéo sans ouvrir de fenêtre. Elle écrit un CSV et un
graphique PNG dans `outputs/` et affiche le nombre total de répétitions :

```bash
python src/analyze_video.py chemin/vers/video.mp4
python src/analyze_video.py chemin/vers/video.mp4 --side left
```

Le CSV contient les colonnes `frame`, `temps`, `angle_genou`, `etat` et
`nb_reps`. Les répétitions détectées sont indiquées sur le graphique.

## Structure

- `src/angles.py` : calcul de l'angle formé par trois points.
- `src/rep_counter.py` : lissage et machine à états à seuils configurables.
- `src/pose_demo.py` : affichage de la pose, de l'angle et du compteur.
- `src/analyze_video.py` : export des mesures et du graphique.
- `data/` et `outputs/` : répertoires réservés aux entrées et résultats.
