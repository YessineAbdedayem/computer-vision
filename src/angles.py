"""Utilitaires de géométrie pour l'analyse de poses."""

import math
from typing import Sequence


def calculate_angle(
    a: Sequence[float],
    b: Sequence[float],
    c: Sequence[float],
) -> float:
    """Retourne l'angle en degrés au point b défini par les points a, b et c."""
    ba_x, ba_y = a[0] - b[0], a[1] - b[1]
    bc_x, bc_y = c[0] - b[0], c[1] - b[1]
    norme_ba = math.hypot(ba_x, ba_y)
    norme_bc = math.hypot(bc_x, bc_y)

    # Un vecteur nul ne définit pas d'angle : on renvoie une valeur exploitable.
    if norme_ba == 0 or norme_bc == 0:
        return 0.0

    produit = ba_x * bc_x + ba_y * bc_y
    cosinus = max(-1.0, min(1.0, produit / (norme_ba * norme_bc)))
    return math.degrees(math.acos(cosinus))
