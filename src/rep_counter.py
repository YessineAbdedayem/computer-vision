"""Comptage des répétitions avec lissage et hystérésis."""

import math
from collections import deque


class RepCounter:
    """Compte les passages de la position basse à la position haute."""

    def __init__(
        self,
        high_threshold: float = 160.0,
        low_threshold: float = 100.0,
        window_size: int = 5,
    ) -> None:
        if low_threshold >= high_threshold:
            raise ValueError("Le seuil bas doit être inférieur au seuil haut.")
        if window_size < 1:
            raise ValueError("La taille de la moyenne mobile doit être positive.")

        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.window_size = window_size
        self._angles: deque[float] = deque(maxlen=window_size)
        self.state: str | None = None
        self.repetitions = 0
        self.smoothed_angle: float | None = None

    @property
    def state_label(self) -> str:
        """Retourne un libellé prêt à afficher, y compris avant l'initialisation."""
        return self.state or "transition"

    def update(self, angle: float) -> bool:
        """Ajoute un angle; retourne True si une répétition vient d'être comptée."""
        if not math.isfinite(angle):
            raise ValueError("L'angle doit être une valeur finie.")

        self._angles.append(float(angle))
        self.smoothed_angle = sum(self._angles) / len(self._angles)
        repetition_terminee = False

        if self.state is None:
            if self.smoothed_angle > self.high_threshold:
                self.state = "haut"
            elif self.smoothed_angle < self.low_threshold:
                self.state = "bas"
        elif self.state == "bas" and self.smoothed_angle > self.high_threshold:
            self.state = "haut"
            self.repetitions += 1
            repetition_terminee = True
        elif self.state == "haut" and self.smoothed_angle < self.low_threshold:
            self.state = "bas"

        return repetition_terminee

    def reset(self) -> None:
        """Réinitialise le lissage, l'état et le nombre de répétitions."""
        self._angles.clear()
        self.state = None
        self.repetitions = 0
        self.smoothed_angle = None
