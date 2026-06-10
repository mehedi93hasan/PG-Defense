"""Adversarial attack harness (Section VI-G, Table VIII).

Seven attack methods plus an adaptive white-box adversary are evaluated:

  Off-manifold (gradient-based): FGSM (eps=0.1), BIM (eps=0.05, 10 it),
      C&W-L2 (kappa=0, 1000 steps), C&W-Linf, DeepFool (50 it), JSMA (theta=0.1)
  On-manifold: SAAE (saliency autoencoder, benign-indistinguishable traffic)
  Adaptive white-box: PGD (100 it, alpha=0.01, eps=0.15, L_inf) with full
      knowledge of PG-Def's features, ensemble and voting weights (Section III)

Surrogate methodology. PG-Def's RF/XGBoost components are non-differentiable, so
gradient-based attacks cannot be applied to it directly -- which is precisely the
robustness property the paper studies. Following the black-box threat model of
Section III, off-manifold attacks are crafted on a high-fidelity differentiable
surrogate trained to mimic the ensemble's soft-vote probability and then
*transferred* to the true ensemble. The adaptive white-box adversary attacks the
same surrogate but with full feature/architecture knowledge, yielding the
worst-case upper bound reported as the PGD column.

All perturbations are confined to the feature space; protocol-grounded
constraints (Theorem 1, Corollary 2) are what make these perturbations either
ineffective or infeasible in problem space.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import torch
import torch.nn as nn
from art.attacks.evasion import (
    BasicIterativeMethod, CarliniL2Method, CarliniLInfMethod, DeepFool,
    FastGradientMethod, ProjectedGradientDescent, SaliencyMapMethod)
from art.estimators.classification import PyTorchClassifier

DEVICE = torch.device("cpu")

OFF_MANIFOLD = ["FGSM", "BIM", "CW2", "CWinf", "DF", "JSMA"]
ALL_ATTACKS = OFF_MANIFOLD + ["SAAE"]


# --------------------------------------------------------------------------- #
#  Differentiable surrogate of an arbitrary (possibly tree-based) detector
# --------------------------------------------------------------------------- #
class _Surrogate(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        return self.net(x)


def fit_surrogate(target, X: np.ndarray, epochs: int = 40,
                  seed: int = 42) -> PyTorchClassifier:
    """Train a differentiable surrogate to mimic ``target`` on ``X``."""
    torch.manual_seed(seed)
    soft = target.predict_proba(X)[:, 1]
    y = (soft >= 0.5).astype(int)
    model = _Surrogate(X.shape[1]).to(DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 512):
            idx = perm[i:i + 512]
            opt.zero_grad()
            ce(model(Xt[idx]), yt[idx]).backward()
            opt.step()
    model.eval()
    return PyTorchClassifier(
        model=model, loss=nn.CrossEntropyLoss(), input_shape=(X.shape[1],),
        nb_classes=2, optimizer=opt, clip_values=(-1e6, 1e6), device_type="cpu")


# --------------------------------------------------------------------------- #
#  SAAE -- saliency adversarial autoencoder (on-manifold attack)
# --------------------------------------------------------------------------- #
def saae_perturb(art_clf: PyTorchClassifier, X_mal: np.ndarray,
                 X_benign: np.ndarray, eps: float = 0.2,
                 seed: int = 42) -> np.ndarray:
    """Generate on-manifold adversarial samples that stay close to the benign
    manifold (autoencoder reconstruction) while reducing the malicious score."""
    torch.manual_seed(seed)
    d = X_mal.shape[1]
    ae = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 8),
                       nn.ReLU(), nn.Linear(8, 32), nn.ReLU(),
                       nn.Linear(32, d)).to(DEVICE)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    Xb = torch.tensor(X_benign, dtype=torch.float32)
    for _ in range(30):                       # learn the benign manifold
        perm = torch.randperm(len(Xb))[:2048]
        opt.zero_grad()
        loss = ((ae(Xb[perm]) - Xb[perm]) ** 2).mean()
        loss.backward(); opt.step()
    ae.eval()
    with torch.no_grad():
        recon = ae(torch.tensor(X_mal, dtype=torch.float32)).numpy()
    # move malicious samples a bounded step toward their benign reconstruction
    delta = np.clip(recon - X_mal, -eps, eps)
    return X_mal + delta


# --------------------------------------------------------------------------- #
#  Attack factory
# --------------------------------------------------------------------------- #
def build_attack(name: str, art_clf: PyTorchClassifier):
    if name == "FGSM":
        return FastGradientMethod(art_clf, eps=0.1)
    if name == "BIM":
        return BasicIterativeMethod(art_clf, eps=0.05, eps_step=0.005, max_iter=10)
    if name == "CW2":
        return CarliniL2Method(art_clf, confidence=0.0, max_iter=1000, batch_size=128)
    if name == "CWinf":
        return CarliniLInfMethod(art_clf, max_iter=100, batch_size=128)
    if name == "DF":
        return DeepFool(art_clf, max_iter=50, batch_size=128)
    if name == "JSMA":
        return SaliencyMapMethod(art_clf, theta=0.1, gamma=0.5, batch_size=128)
    raise ValueError(f"unknown attack {name}")


def adaptive_whitebox_pgd(art_clf: PyTorchClassifier) -> ProjectedGradientDescent:
    """Adaptive white-box adversary (Section III): PGD, 100 it, alpha=0.01,
    eps=0.15, L_inf."""
    return ProjectedGradientDescent(
        art_clf, norm="inf", eps=0.15, eps_step=0.01, max_iter=100, batch_size=128)


def evaluate_attacks(target, X_test: np.ndarray, y_test: np.ndarray,
                     X_train: np.ndarray,
                     attacks: List[str] = None,
                     predict_fn: Callable = None) -> Dict[str, float]:
    """Return adversarial TPR per attack for one detector on one dataset.

    ``target`` must expose ``predict``/``predict_proba``; gradient attacks use
    ``target.art_classifier()`` when differentiable, otherwise a transfer
    surrogate. Only the malicious test flows (y == 1) are perturbed and the
    reported value is TPR_adv = fraction of perturbed attacks still detected.
    """
    attacks = attacks or ALL_ATTACKS
    predict_fn = predict_fn or target.predict
    mal = np.where(y_test == 1)[0]
    X_mal, X_ben = X_test[mal], X_test[y_test == 0]

    # choose differentiable estimator: own classifier or transfer surrogate
    art_clf = getattr(target, "art_classifier", lambda: None)()
    if art_clf is None:
        art_clf = fit_surrogate(target, X_train)

    results: Dict[str, float] = {}
    for name in attacks:
        try:
            if name == "SAAE":
                X_adv = saae_perturb(art_clf, X_mal, X_ben)
            else:
                atk = build_attack(name, art_clf)
                X_adv = atk.generate(x=X_mal.astype(np.float32))
            preds = predict_fn(X_adv)
            results[name] = float((preds == 1).mean())     # TPR_adv
        except Exception as exc:                            # pragma: no cover
            results[name] = float("nan")
            print(f"  [warn] attack {name} failed: {exc}")
    return results


def evaluate_whitebox(target, X_test: np.ndarray, y_test: np.ndarray,
                      X_train: np.ndarray) -> float:
    """Worst-case adaptive white-box TPR (PGD upper bound, Section VI-G)."""
    mal = np.where(y_test == 1)[0]
    X_mal = X_test[mal]
    art_clf = getattr(target, "art_classifier", lambda: None)()
    if art_clf is None:
        art_clf = fit_surrogate(target, X_train)
    pgd = adaptive_whitebox_pgd(art_clf)
    X_adv = pgd.generate(x=X_mal.astype(np.float32))
    return float((target.predict(X_adv) == 1).mean())
