"""Attack-every-point helper (2026-10-16 switch; see report/ATTACK_ALL_PLAN.md).

Every test point is attacked from the model's own prediction (as in Aidan's original code);
only candidates whose prediction changed are kept. Restricting the attack to correctly
classified points would remove exactly the points a defect damages most and would make
defect and control arms measure different sets of points (Johnson's fixed-cohort check).

DecisionTreeAttack is run ONE POINT AT A TIME so that ART's IndexError (its upward search
overwrites earlier hits and runs past the root; analysis/verify_dta_destination.py) costs
one point, not the whole fold. A batched call would lose the entire fold on the first such
point, and the bug is more likely under heavy defects.
"""
import numpy as np
from art.attacks.evasion import DecisionTreeAttack


def attack_all(art, Xv):
    """Attack every row of Xv with DecisionTreeAttack, one point at a time.
    Returns (candidates, number of points ART could not attack)."""
    atk = DecisionTreeAttack(classifier=art)
    out, err = Xv.copy(), 0
    for i in range(len(Xv)):
        try:
            out[i] = atk.generate(Xv[i:i + 1])[0]
        except IndexError:
            err += 1
    return out, err


def flips(art, cand, pv):
    """Boolean mask: the candidate's prediction differs from the model's prediction on the input."""
    return np.argmax(art.predict(cand), axis=1) != pv
