"""Realizable adversarial attacks confined to the manipulable feature subspace.

All attacks perturb ONLY the manipulable feature indices, leaving protected
features (and thus attack behaviour) unchanged -- the realizability constraint of
the manuscript. A detector trained on protected features is therefore inert to
these perturbations by construction.
"""
import numpy as np


def mimicry(X, y, manip_idx, benign_mean):
    """Set the manipulable features of every attack row to the benign centroid.
    The cheapest realizable attack (a static header/padding template)."""
    Xa = X.copy()
    atk = y == 1
    for j in manip_idx:
        Xa[atk, j] = benign_mean[j]
    return Xa


def _logreg_surrogate(model, X):
    """Numerical gradient of the ensemble score wrt inputs (model-agnostic),
    used to drive FGSM/BIM/PGD without assuming differentiability."""
    eps = 1e-3
    base = model.proba(X)
    grad = np.zeros_like(X)
    for j in range(X.shape[1]):
        Xp = X.copy(); Xp[:, j] += eps
        grad[:, j] = (model.proba(Xp) - base) / eps
    return grad


def _project(Xa, X0, manip_idx, eps, sigma):
    """Keep perturbation within an Linf eps-ball (in sigma units) and on the
    manipulable axes only."""
    mask = np.zeros(X0.shape[1], dtype=bool); mask[list(manip_idx)] = True
    delta = (Xa - X0)
    delta[:, ~mask] = 0.0
    bound = eps * sigma
    delta = np.clip(delta, -bound, bound)
    return X0 + delta


def gradient_attack(model, X, y, manip_idx, sigma, eps=0.15, steps=1,
                    random_start=False, seed=0):
    """Unified FGSM (steps=1) / BIM (steps>1) / PGD (steps>1, random_start=True)
    that descends the attack-class score, projected onto the manipulable subspace."""
    rng = np.random.default_rng(seed)
    X0 = X.copy(); Xa = X.copy()
    atk = y == 1
    if random_start:
        for j in manip_idx:
            Xa[atk, j] += rng.uniform(-eps * sigma[j], eps * sigma[j], atk.sum())
        Xa = _project(Xa, X0, manip_idx, eps, sigma)
    step = (eps / max(steps, 1)) * sigma
    for _ in range(steps):
        g = _logreg_surrogate(model, Xa)
        Xa[atk] = Xa[atk] - (step * np.sign(g))[atk]      # lower the malicious score
        Xa = _project(Xa, X0, manip_idx, eps, sigma)
    return Xa
